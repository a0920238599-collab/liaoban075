package com.realtek.chat.ai

import android.content.Context
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.realtek.chat.settings.AppSettings
import com.realtek.chat.storage.*
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.withTimeoutOrNull
import java.time.OffsetDateTime
import java.time.format.DateTimeParseException


private data class ShortContinuationDecision(
    val shouldContinue: Boolean,
    val turns: Int,
    val direction: String
)

class ChatEngine(context: Context) {
    private val appContext = context.applicationContext
    private val db = AppDb.get(appContext)
    private val gateway = RunApiGateway(appContext)
    private val contextEngine = ContextEngine(appContext)
    private val settings = AppSettings(appContext)
    private val media = SecureMediaStore(appContext)
    private val planner = ConversationPlanner(appContext)
    private val socialEngine = SocialEngine(appContext)
    private val repetitionGuard = RepetitionGuard(
        appContext,
        socialEngine
    )

    suspend fun sendText(
        contactId: Int,
        text: String,
        onChanged: () -> Unit = {},
        onStreaming: (String) -> Unit = {}
    ) {
        val clean = text.trim()
        if (clean.isBlank()) return

        val contact = db.getContact(contactId) ?: return
        val triggerMessageId = db.addMessage(
            contactId,
            "user",
            "text",
            clean
        )
        onChanged()

        val fastMode = isFastCasualMessage(clean)

        val initialSocialState =
            socialEngine.prepareState(
                contact,
                clean
            )

        val plan = planner.plan(
            contact = contact,
            currentText = clean,
            fastMode = fastMode
        )

        val socialState =
            socialEngine.applyPlan(
                initialSocialState,
                plan
            )

        val selectedMemories = planner.resolveMemories(
            contact.id,
            plan
        )

        runStages(
            contact = contact,
            triggerMessageId = triggerMessageId,
            currentText = clean,
            fastMode = fastMode,
            plan = plan,
            selectedMemories = selectedMemories,
            socialState = socialState,
            onChanged = onChanged,
            onStreaming = onStreaming
        )

        // talk_burst 本身已经是连续发言，不再叠加短续聊。
        if (
            plan.mode != "talk_burst" &&
            plan.maxStages <= 1 &&
            !settings.sleepMode &&
            contact.proactiveEnabled
        ) {
            maybeShortContinue(
                contact = contact,
                originalUserText = clean,
                plan = plan,
                socialState = socialState,
                onChanged = onChanged,
                onStreaming = onStreaming
            )
        }
    }

    private suspend fun runStages(
        contact: Contact,
        triggerMessageId: Long,
        currentText: String,
        fastMode: Boolean,
        plan: ConversationPlan,
        selectedMemories: List<MemoryItem>,
        socialState: SocialState,
        onChanged: () -> Unit,
        onStreaming: (String) -> Unit
    ) {
        var expectedLastMessageId = triggerMessageId
        var previousStageText: String? = null
        var completedStages = 0

        for (stageIndex in 0 until plan.maxStages) {
            if (stageIndex > 0) {
                val pause = (
                    plan.pauseMs +
                        stableJitter(expectedLastMessageId, 360L)
                    ).coerceIn(350L, 2600L)

                delay(pause)

                // 用户一旦插话，旧的话轮立刻让出，不再继续“抢着说”。
                val beforeContinue = db.getLastMessage(contact.id)
                    ?: break

                if (
                    beforeContinue.id != expectedLastMessageId ||
                    beforeContinue.role != "assistant"
                ) {
                    break
                }

                val continueSpeaking = planner.shouldContinue(
                    contact = contact,
                    plan = plan,
                    completedStages = completedStages
                )

                if (!continueSpeaking) break

                // planner 判断期间用户也可能插话，再检查一次。
                val afterPlanner = db.getLastMessage(contact.id)
                    ?: break

                if (
                    afterPlanner.id != expectedLastMessageId ||
                    afterPlanner.role != "assistant"
                ) {
                    break
                }
            } else {
                val last = db.getLastMessage(contact.id)
                    ?: return

                if (
                    last.id != triggerMessageId ||
                    last.role != "user"
                ) {
                    return
                }
            }

            val bundle = contextEngine.build(
                contact = contact,
                currentText = currentText,
                fastMode = fastMode,
                plan = plan,
                selectedMemories = selectedMemories,
                socialState = socialState,
                recentSelfTopics =
                    socialEngine.recentTopicSummary(
                        contact.id
                    ),
                recentAssistantText =
                    repetitionGuard.recentAssistantText(
                        contact.id
                    ),
                stageIndex = stageIndex,
                previousStageText = previousStageText
            )

            val selectedModel =
                if (
                    fastMode &&
                    settings.fastChatModel.isNotBlank()
                ) {
                    settings.fastChatModel
                } else {
                    contact.model
                }

            val maxTokens = when {
                plan.mode == "talk_burst" -> 260
                fastMode -> 180
                plan.mode == "backchannel" -> 90
                else -> 520
            }

            val generated = generateGuardedStage(
                contact = contact,
                model = selectedModel,
                bundle = bundle,
                maxTokens = maxTokens,
                onStreaming = onStreaming
            )

            if (
                generated == null ||
                generated.isBlank()
            ) {
                onStreaming("")
                break
            }

            val output = generated

            // 流式生成期间如果用户已经插话，这一段作废，不写入数据库。
            val currentLast = db.getLastMessage(contact.id)
                ?: break

            val expectedRole =
                if (stageIndex == 0) "user" else "assistant"

            if (
                currentLast.id != expectedLastMessageId ||
                currentLast.role != expectedRole
            ) {
                onStreaming("")
                break
            }

            val assistantId = db.addMessage(
                contact.id,
                "assistant",
                "text",
                output.take(2200)
            )

            expectedLastMessageId = assistantId
            previousStageText = output
            completedStages += 1

            socialEngine.afterAssistantMessage(
                contact.id,
                output
            )

            onStreaming("")
            onChanged()
        }

        // v7.4 不再安排 45~135 秒的 WorkManager 跟进。
        // 正在聊天时的续聊由当前前台会话中的 2~5 秒短续聊处理。
    }


private suspend fun maybeShortContinue(
    contact: Contact,
    originalUserText: String,
    plan: ConversationPlan,
    socialState: SocialState,
    onChanged: () -> Unit,
    onStreaming: (String) -> Unit
) = coroutineScope {
    val firstLast =
        db.getLastMessage(contact.id)
            ?: return@coroutineScope

    if (firstLast.role != "assistant") {
        return@coroutineScope
    }

    val explicitContinue =
        isExplicitContinueRequest(
            originalUserText
        )

    val maxExtraTurns =
        if (explicitContinue) {
            3
        } else {
            1
        }

    // 先用本地节奏控制器决定“这一轮值不值得让 DeepSeek 再判断”。
    // 普通一句话大多数不会进入续聊，避免固定成“你一句、AI 三句”。
    if (
        !shouldOfferShortContinuation(
            contact = contact,
            userText = originalUserText,
            plan = plan,
            state = socialState,
            seed = firstLast.id
        )
    ) {
        return@coroutineScope
    }

    val expectedAtDecision =
        firstLast.id

    // 决策和自然停顿并行，避免先等几秒再开始问 DeepSeek。
    val decisionDeferred = async {
        withTimeoutOrNull(
            3_500L
        ) {
            decideShortContinuation(
                contact = contact,
                originalUserText =
                    originalUserText,
                maxExtraTurns =
                    maxExtraTurns,
                explicitContinue =
                    explicitContinue
            )
        } ?: ShortContinuationDecision(
            shouldContinue = false,
            turns = 0,
            direction = ""
        )
    }

    delay(
        shortContinuationPause(
            contact = contact,
            state = socialState,
            previousText = firstLast.text,
            seed = expectedAtDecision
        )
    )

    // 这里检查的是“用户是否真的已经发出新消息”，不是输入框检测。
    val afterPause =
        db.getLastMessage(contact.id)
            ?: return@coroutineScope

    if (
        afterPause.id != expectedAtDecision ||
        afterPause.role != "assistant"
    ) {
        return@coroutineScope
    }

    val decision =
        decisionDeferred.await()

    if (
        !decision.shouldContinue ||
        decision.turns <= 0
    ) {
        return@coroutineScope
    }

    val actualTurns =
        decision.turns.coerceIn(
            1,
            maxExtraTurns
        )

    var expectedLastMessageId =
        expectedAtDecision
    var previousContinuation: String? =
        null

    for (
        index in 0 until actualTurns
    ) {
        if (index > 0) {
            delay(
                shortContinuationPause(
                    contact = contact,
                    state = socialState,
                    previousText =
                        previousContinuation
                            .orEmpty(),
                    seed =
                        expectedLastMessageId +
                            index * 31L
                )
            )
        }

        val beforeGenerate =
            db.getLastMessage(contact.id)
                ?: break

        if (
            beforeGenerate.id !=
                expectedLastMessageId ||
            beforeGenerate.role !=
                "assistant"
        ) {
            break
        }

        val bundle =
            buildShortContinuationBundle(
                contact = contact,
                originalUserText =
                    originalUserText,
                direction =
                    decision.direction,
                index = index,
                totalTurns =
                    actualTurns,
                previousContinuation =
                    previousContinuation
            )

        val generated =
            generateShortContinuationSafely(
                contact = contact,
                expectedLastMessageId =
                    expectedLastMessageId,
                bundle = bundle,
                onStreaming =
                    onStreaming
            )
                ?: break

        // 生成期间用户发了新消息，旧续聊立即作废。
        val beforeSave =
            db.getLastMessage(contact.id)
                ?: break

        if (
            beforeSave.id !=
                expectedLastMessageId ||
            beforeSave.role !=
                "assistant"
        ) {
            onStreaming("")
            break
        }

        val assistantId =
            db.addMessage(
                contact.id,
                "assistant",
                "text",
                generated.take(1200)
            )

        expectedLastMessageId =
            assistantId
        previousContinuation =
            generated

        socialEngine
            .afterAssistantMessage(
                contact.id,
                generated
            )

        onStreaming("")
        onChanged()
    }
}

private suspend fun decideShortContinuation(
    contact: Contact,
    originalUserText: String,
    maxExtraTurns: Int,
    explicitContinue: Boolean
): ShortContinuationDecision {
    if (
        !settings.providerConfigured(
            settings.systemProvider
        )
    ) {
        return ShortContinuationDecision(
            shouldContinue = false,
            turns = 0,
            direction = ""
        )
    }

    val recent =
        db.getRecentMessages(
            contact.id,
            10
        )

    val transcript =
        recent.joinToString("\n") {
            "${if (it.role == "user") "用户" else contact.name}：" +
                if (
                    it.type == "image"
                ) {
                    "[图片]"
                } else {
                    it.text
                }
        }

    val state =
        socialEngine.snapshot(
            contact.id
        )

    val raw = runCatching {
        gateway.chat(
            provider =
                settings.systemProvider,
            model =
                settings.defaultModelFor(
                    settings.systemProvider
                ),
            systemPrompt = """
                你只负责即时聊天的话轮判断，不负责写最终聊天内容。

                核心原则：
                1. 默认是“不要继续”。普通闲聊大多数时候一条回复就够。
                2. 只有当前话题确实还有自然的话没说完，才 continue=true。
                3. 不要为了活跃、热情或怕冷场而硬续聊。
                4. 已经回答完整的问题、话题自然结束、继续只会重复时，必须停止。
                5. 用户在分享事情、情绪明显、只给短反馈而话题仍在进行时，可以更愿意补一句。
                6. 普通场景最多再补 1 条；只有用户明确要求“继续说/多说点”时才允许多条。
                7. direction 只描述继续聊的方向，不写具体台词。
                8. 优先延续当前话题，不要突然换新话题。
            """.trimIndent(),
            userPrompt = """
                联系人：${contact.name}

                用户这一轮原始消息：
                $originalUserText

                用户是否明确要求继续说：
                $explicitContinue

                本轮最多允许追加：
                $maxExtraTurns 条

                当前聊天状态：
                mood=${state.mood}
                topic=${state.currentTopic}
                phase=${state.phase}
                familiarity=${"%.2f".format(state.familiarity)}
                social_drive=${"%.2f".format(state.socialDrive)}

                最近聊天：
                $transcript

                最近联系人自我话题：
                ${socialEngine.recentTopicSummary(contact.id)}

                只输出 JSON：
                {
                  "continue": true,
                  "turns": 1,
                  "direction": "沿当前话题自然补充一点"
                }

                或：
                {
                  "continue": false,
                  "turns": 0,
                  "direction": ""
                }

                turns 不能超过 $maxExtraTurns。
            """.trimIndent(),
            maxTokens = 110
        )
    }.getOrNull()
        ?: return ShortContinuationDecision(
            false,
            0,
            ""
        )

    val obj =
        parseJson(raw)
            ?: return ShortContinuationDecision(
                false,
                0,
                ""
            )

    val shouldContinue =
        obj["continue"]
            ?.takeIf {
                it.isJsonPrimitive
            }
            ?.asBoolean
            ?: false

    if (!shouldContinue) {
        return ShortContinuationDecision(
            false,
            0,
            ""
        )
    }

    return ShortContinuationDecision(
        shouldContinue = true,
        turns =
            (
                obj["turns"]
                    ?.takeIf {
                        it.isJsonPrimitive
                    }
                    ?.asInt
                    ?: 1
                ).coerceIn(
                1,
                maxExtraTurns
            ),
        direction =
            obj["direction"]
                ?.takeIf {
                    it.isJsonPrimitive
                }
                ?.asString
                ?.trim()
                .orEmpty()
                .take(180)
    )
}

private suspend fun generateShortContinuationSafely(
    contact: Contact,
    expectedLastMessageId: Long,
    bundle: PromptBundle,
    onStreaming: (String) -> Unit
): String? {
    val first =
        runCatching {
            generateGuardedStage(
                contact = contact,
                model = contact.model,
                bundle = bundle,
                maxTokens = 220,
                onStreaming =
                    onStreaming
            )
        }.getOrNull()

    if (
        first != null &&
        first.isNotBlank()
    ) {
        return first
    }

    // 续聊失败不向用户弹大红错误；清掉残留流式文本后静默重试一次。
    onStreaming("")

    delay(
        450L +
            stableJitter(
                expectedLastMessageId,
                350L
            )
    )

    val last =
        db.getLastMessage(contact.id)
            ?: return null

    if (
        last.id != expectedLastMessageId ||
        last.role != "assistant"
    ) {
        return null
    }

    val retry =
        runCatching {
            generateGuardedStage(
                contact = contact,
                model = contact.model,
                bundle = bundle,
                maxTokens = 220,
                onStreaming =
                    onStreaming
            )
        }.getOrNull()

    if (
        retry.isNullOrBlank()
    ) {
        onStreaming("")
        return null
    }

    return retry
}

private fun shouldOfferShortContinuation(
    contact: Contact,
    userText: String,
    plan: ConversationPlan,
    state: SocialState,
    seed: Long
): Boolean {
    val clean =
        userText.trim()

    if (
        isExplicitContinueRequest(
            clean
        )
    ) {
        return true
    }

    val closingSignals =
        listOf(
            "好的",
            "好吧",
            "行",
            "知道了",
            "明白了",
            "谢谢",
            "晚安",
            "拜拜",
            "先这样",
            "嗯好"
        )

    if (
        closingSignals.any {
            clean == it ||
                clean.endsWith(it)
        }
    ) {
        return false
    }

    val backchannelSignals =
        listOf(
            "嗯",
            "嗯嗯",
            "啊",
            "哦",
            "哈哈",
            "哈哈哈",
            "真的",
            "是吗",
            "然后呢",
            "后来呢",
            "真的假的"
        )

    var chance =
        when {
            backchannelSignals.any {
                clean == it ||
                    clean.contains(it)
            } -> 0.66

            plan.mode == "react" ->
                0.55

            plan.mode ==
                "listen_then_respond" ->
                0.60

            plan.mode == "casual" ->
                0.38

            plan.mode == "answer" ->
                0.14

            plan.mode == "repair" ->
                0.10

            else ->
                0.24
        }

    // 越熟、越愿意交流，补一句的概率略高，但幅度有限。
    chance +=
        (state.familiarity - 0.5) *
            0.16

    chance +=
        (state.socialDrive - 0.5) *
            0.10

    chance +=
        contactRhythmBias(
            contact
        )

    chance =
        chance.coerceIn(
            0.06,
            0.76
        )

    return stableFraction(
        seed +
            contact.id * 131L +
            clean.hashCode().toLong()
    ) < chance
}

private fun isExplicitContinueRequest(
    text: String
): Boolean {
    val signals =
        listOf(
            "继续说",
            "继续讲",
            "接着说",
            "接着讲",
            "多说点",
            "多聊点",
            "你说吧",
            "你继续",
            "再说点",
            "讲下去",
            "我听着",
            "你多讲一会",
            "陪我聊会儿",
            "你随便说"
        )

    return signals.any {
        text.contains(it)
    }
}

private fun contactRhythmBias(
    contact: Contact
): Double {
    val style =
        (
            contact.corePersona +
                " " +
                contact.styleRules
            ).lowercase()

    var bias = 0.0

    if (
        listOf(
            "话少",
            "简短",
            "克制",
            "慢热",
            "不多话"
        ).any {
            style.contains(it)
        }
    ) {
        bias -= 0.08
    }

    if (
        listOf(
            "健谈",
            "话多",
            "爱聊",
            "活泼",
            "主动",
            "调侃"
        ).any {
            style.contains(it)
        }
    ) {
        bias += 0.08
    }

    // 给不同联系人一点稳定但不明显的节奏差异。
    val stable =
        (
            (
                contact.id * 37L +
                    contact.name.hashCode()
                ) and
                0x7fffffffL
            ) % 17L

    bias +=
        (
            stable.toDouble() -
                8.0
            ) / 180.0

    return bias
}

private fun buildShortContinuationBundle(
    contact: Contact,
    originalUserText: String,
    direction: String,
    index: Int,
    totalTurns: Int,
    previousContinuation: String?
): PromptBundle {
    val recent =
        db.getRecentMessages(
            contact.id,
            10
        )

    val transcript =
        recent.joinToString("\n") {
            "${if (it.role == "user") "用户" else contact.name}：" +
                if (
                    it.type == "image"
                ) {
                    "[图片]"
                } else {
                    it.text
                }
        }

    val state =
        socialEngine.snapshot(
            contact.id
        )

    val system = """
        你正在作为联系人“${contact.name}”进行即时聊天。

        核心性格：
        ${contact.corePersona}

        说话方式：
        ${contact.styleRules}

        当前关系与聊天状态：
        mood=${state.mood}
        current_topic=${state.currentTopic}
        phase=${state.phase}
        familiarity=${"%.2f".format(state.familiarity)}

        现在不是重新开始一轮问答，而是上一句话之后自然地再补一点。

        继续方向：
        ${direction.ifBlank { "沿当前话题自然往前推进一点" }}

        规则：
        1. 优先延续当前话题，不要无缘无故换话题。
        2. 只发一条自然消息，不把完整答案拆成多条。
        3. 不重复刚说过的话，也不要换个说法复述。
        4. 不强行追问，不要每条都以问题结尾。
        5. 可以补充一个新细节、一个反应、一个联想或一句调侃。
        6. 如果这是普通续聊，短一点通常更自然。
        7. 不写列表，不写解释性前言。
        8. 不虚构现实线下身体、所在地、工作、吃饭、睡觉等经历。
    """.trimIndent()

    val user = """
        用户这一轮最初说：
        $originalUserText

        最近聊天：
        $transcript

        这是本轮自然续聊中的第 ${index + 1}/$totalTurns 条。

        ${previousContinuation?.let {
            "你上一条刚说：$it\n继续时必须推进一点新内容，不要重复。"
        }.orEmpty()}

        只输出这一条聊天消息本身。
    """.trimIndent()

    return PromptBundle(
        system = system,
        user = user
    )
}

private fun shortContinuationPause(
    contact: Contact,
    state: SocialState,
    previousText: String,
    seed: Long
): Long {
    val length =
        previousText.trim()
            .length

    var base =
        when {
            length <= 8 -> 1350L
            length <= 28 -> 1850L
            length <= 70 -> 2350L
            else -> 2850L
        }

    val style =
        (
            contact.corePersona +
                " " +
                contact.styleRules
            ).lowercase()

    if (
        listOf(
            "慢热",
            "克制",
            "沉稳"
        ).any {
            style.contains(it)
        }
    ) {
        base += 250L
    }

    if (
        listOf(
            "反应快",
            "活泼",
            "健谈",
            "调侃"
        ).any {
            style.contains(it)
        }
    ) {
        base -= 180L
    }

    base -=
        (
            state.familiarity *
                180.0
            ).toLong()

    return (
        base +
            stableJitter(
                seed,
                850L
            )
        ).coerceIn(
        1100L,
        3800L
    )
}

private fun stableFraction(
    seed: Long
): Double {
    var x =
        seed * 1103515245L +
            12345L

    x =
        x xor
            (x ushr 21)

    x =
        x xor
            (x shl 13)

    x =
        x xor
            (x ushr 7)

    val positive =
        x and
            0x7fffffffffffffffL

    return (
        positive % 10_000L
        ).toDouble() /
        10_000.0
}

    suspend fun sendImage(
        contactId: Int,
        mediaPath: String,
        onChanged: () -> Unit = {}
    ) {
        val contact = db.getContact(contactId) ?: return

        db.addMessage(
            contactId,
            "user",
            "image",
            "[图片]",
            mediaPath,
            "image/jpeg"
        )
        onChanged()

        val bytes = media.loadBytes(mediaPath)

        val raw = runCatching {
            gateway.chat(
                provider = contact.provider,
                model = contact.model,
                systemPrompt = """
                    你是联系人 ${contact.name}。
                    核心性格：${contact.corePersona}
                    说话方式：${contact.styleRules}
                    用户刚发了一张图片。像即时聊天联系人一样自然回应，不要自动写成图片分析报告。
                """.trimIndent(),
                userPrompt = "用户刚刚发送了一张图片，请结合图片自然回应。",
                imageBytes = bytes,
                imageMime = "image/jpeg",
                maxTokens = 260
            )
        }.getOrElse {
            gateway.chat(
                provider = contact.provider,
                model = contact.model,
                systemPrompt = """
                    你是联系人 ${contact.name}。
                    核心性格：${contact.corePersona}
                    说话方式：${contact.styleRules}
                """.trimIndent(),
                userPrompt = "用户刚刚发了一张图片，但当前模型未能读取图片。自然回应一下，不要假装看到了具体内容。",
                maxTokens = 160
            )
        }

        val imageReply =
            raw.trim().take(1200)

        db.addMessage(
            contact.id,
            "assistant",
            "text",
            imageReply
        )

        socialEngine.afterAssistantMessage(
            contact.id,
            imageReply
        )

        onChanged()
    }

    suspend fun generatePersona(
        name: String,
        description: String
    ): GeneratedPersona {
        require(
            settings.providerConfigured(
                settings.systemProvider
            )
        ) {
            "请先到“我 → 设置 → AI 服务”配置系统后台所使用的 API。"
        }

        val raw = gateway.chat(
            provider = settings.systemProvider,
            model = settings.defaultModelFor(
                settings.systemProvider
            ),
            systemPrompt = """
                你是人物设定编辑器。
                根据名字和用户的简单描述，生成一个长期稳定、适合即时聊天的人物设定。
                不写小说背景，重点是核心性格和自然聊天习惯。
            """.trimIndent(),
            userPrompt = """
                名字：${name.trim()}
                描述：${description.trim()}

                只输出 JSON：
                {
                  "subtitle":"不超过20字的简介",
                  "persona":"完整核心性格",
                  "style":"聊天表达习惯"
                }
            """.trimIndent(),
            maxTokens = 320
        )

        val obj = parseJson(raw)
            ?: error("人物设定生成失败，请再试一次。")

        return GeneratedPersona(
            subtitle = obj["subtitle"]
                ?.asString
                ?.trim()
                .orEmpty(),
            persona = obj["persona"]
                ?.asString
                ?.trim()
                .orEmpty(),
            style = obj["style"]
                ?.asString
                ?.trim()
                .orEmpty()
        )
    }

    suspend fun processMemory(contactId: Int) {
        val contact = db.getContact(contactId) ?: return
        val count = db.userMessageCount(contactId)

        if (
            !settings.providerConfigured(
                settings.systemProvider
            ) ||
            count == 0 ||
            count % 5 != 0
        ) {
            return
        }

        val recent = db.getRecentMessages(contactId, 16)
        val oldSummary = db.getSummary(contactId)
            ?.summary
            .orEmpty()

        val transcript = recent.joinToString("\n") {
            "${if (it.role == "user") "用户" else contact.name}：" +
                if (it.type == "image") "[图片]" else it.text
        }

        val raw = runCatching {
            gateway.chat(
                provider = settings.systemProvider,
                model = settings.memoryModel.ifBlank {
                    settings.defaultModelFor(
                        settings.systemProvider
                    )
                },
                systemPrompt = """
                    你是聊天记忆整理器。
                    目标不是尽量多记，而是保留以后真正能帮助“理解上下文”的内容。
                    特别关注：
                    - 用户明确偏好、人物关系、习惯；
                    - 未来事件；
                    - 尚未结束/等待后续结果的话题（type=open_loop）；
                    - 共同经历；
                    - 用户习惯使用的替代表达。
                    - 只把“用户说过/用户经历/共同聊天中确实发生”的内容写进长期记忆。
                    - AI 联系人自己的临时自述（例如“我累了、我困了、我无聊”）不要提取为用户记忆，也不要写成长期事实。

                    tags 不只是关键词，还要包含可能的同义说法、代称和以后可能触发这段记忆的表达。
                    不记录密码、API Key 等秘密。
                """.trimIndent(),
                userPrompt = """
                    旧摘要：
                    ${oldSummary.ifBlank { "暂无" }}

                    最近聊天：
                    $transcript

                    只输出 JSON：
                    {
                      "summary":"不超过160字的新摘要",
                      "memories":[
                        {
                          "type":"preference/event/person/habit/shared_history/communication_style/open_loop",
                          "content":"一条独立可读的记忆",
                          "tags":"关键词、同义说法、代称、可能触发表达",
                          "importance":0.0,
                          "confidence":0.0,
                          "due_at":null
                        }
                      ]
                    }
                """.trimIndent(),
                maxTokens = 520
            )
        }.getOrNull() ?: return

        val obj = parseJson(raw) ?: return

        obj["summary"]
            ?.takeIf { it.isJsonPrimitive }
            ?.asString
            ?.trim()
            ?.takeIf { it.isNotBlank() }
            ?.let {
                db.upsertSummary(
                    contactId,
                    it.take(600)
                )
            }

        val arr = obj.getAsJsonArray("memories")
            ?: return

        for (i in 0 until minOf(arr.size(), 5)) {
            val el = arr[i]
            if (!el.isJsonObject) continue

            val mo = el.asJsonObject
            val content = mo["content"]
                ?.takeIf { it.isJsonPrimitive }
                ?.asString
                ?.trim()
                .orEmpty()

            if (
                content.length < 2 ||
                db.memoryExists(contactId, content)
            ) {
                continue
            }

            val importance = mo["importance"]
                ?.takeIf { it.isJsonPrimitive }
                ?.asDouble
                ?.coerceIn(0.0, 1.0)
                ?: 0.5

            val confidence = mo["confidence"]
                ?.takeIf { it.isJsonPrimitive }
                ?.asDouble
                ?.coerceIn(0.0, 1.0)
                ?: 0.7

            if (importance < 0.35 || confidence < 0.5) {
                continue
            }

            val dueAt = mo["due_at"]
                ?.takeIf {
                    it.isJsonPrimitive &&
                        !it.asString.equals("null", true)
                }
                ?.asString
                ?.let(::parseTime)

            db.addMemory(
                contactId = contactId,
                type = mo["type"]
                    ?.takeIf { it.isJsonPrimitive }
                    ?.asString
                    ?: "event",
                content = content.take(360),
                tags = mo["tags"]
                    ?.takeIf { it.isJsonPrimitive }
                    ?.asString
                    .orEmpty()
                    .take(200),
                importance = importance,
                confidence = confidence,
                dueAt = dueAt
            )
        }
    }

    suspend fun maybeProactive(
        contact: Contact,
        reason: String,
        conversationContinuation: Boolean = false
    ): String? {
        if (settings.sleepMode) return null
        if (
            !settings.providerConfigured(
                settings.systemProvider
            )
        ) return null

        val recent = db.getRecentMessages(contact.id, 10)
        val summary = db.getSummary(contact.id)
            ?.summary
            .orEmpty()

        val memories = db.getMemories(contact.id, 100)
            .sortedWith(
                compareByDescending<MemoryItem> {
                    it.type == "open_loop"
                }.thenByDescending {
                    it.importance
                }
            )
            .take(6)

        val transcript = recent.joinToString("\n") {
            "${if (it.role == "user") "用户" else contact.name}：" +
                if (it.type == "image") "[图片]" else it.text
        }

        val socialState =
            socialEngine.snapshot(
                contact.id
            )

        val recentSelfTopics =
            socialEngine.recentTopicSummary(
                contact.id
            )

        val recentAssistantText =
            repetitionGuard.recentAssistantText(
                contact.id
            )

        val raw = runCatching {
            gateway.chat(
                provider = settings.systemProvider,
                model = settings.proactiveModel.ifBlank {
                    settings.defaultModelFor(
                        settings.systemProvider
                    )
                },
                systemPrompt = """
                    你是联系人 ${contact.name}。
                    核心性格：${contact.corePersona}
                    说话方式：${contact.styleRules}

                    你不是等用户提问才工作的问答助手。
                    可以自然地主动补一句或重新开启话题，
                    但不要机械刷存在感，不要像客服回访。

                    当前聊天行为状态：
                    mood=${socialState.mood}
                    social_drive=${"%.2f".format(socialState.socialDrive)}
                    familiarity=${"%.2f".format(socialState.familiarity)}
                    current_topic=${socialState.currentTopic.ifBlank { "暂无" }}

                    注意：这些是聊天行为状态，不是现实身体状态。
                    不要虚构自己刚下班、刚吃饭、身体很累、正在某个地点等现实经历。
                """.trimIndent(),
                userPrompt = """
                    触发原因：
                    $reason

                    类型：
                    ${if (conversationContinuation) "刚才对话的自然延续" else "隔一段时间后的主动联系"}

                    近期摘要：
                    ${summary.ifBlank { "暂无" }}

                    相关记忆：
                    ${memories.joinToString("\n") { "- ${it.content}" }.ifBlank { "暂无" }}

                    最近聊天：
                    ${transcript.ifBlank { "暂无" }}

                    你最近已经主动/自述过的主题：
                    $recentSelfTopics

                    你最近自己已经说过的话：
                    $recentAssistantText

                    重要：
                    - 不要再次用最近已经说过的“累、困、无聊、天气”等自我状态开场。
                    - 没有新的、自然的内容时，宁可 send=false。
                    - 如果要发，必须带来新话题、新进展或对未完成话题的自然承接。

                    只输出 JSON：
                    {
                      "send":true,
                      "messages":["第一条","可选第二条"]
                    }
                    或
                    {
                      "send":false,
                      "messages":[]
                    }

                    最多两条短消息。
                """.trimIndent(),
                maxTokens = 180
            )
        }.getOrNull() ?: return null

        val obj = parseJson(raw) ?: return null
        val send = obj["send"]
            ?.takeIf { it.isJsonPrimitive }
            ?.asBoolean
            ?: false

        val arr = obj.getAsJsonArray("messages")
        if (!send || arr == null || arr.size() == 0) {
            return null
        }

        val sent = mutableListOf<String>()

        for (i in 0 until minOf(arr.size(), 2)) {
            val text = arr[i].asString.trim()

            if (
                text.isBlank() ||
                repetitionGuard.isRepetitive(
                    contact.id,
                    text
                )
            ) {
                continue
            }

            if (i > 0) {
                delay(
                    naturalDelay(
                        text,
                        i
                    )
                )
            }

            db.addMessage(
                contact.id,
                "assistant",
                "text",
                text.take(600)
            )

            socialEngine.afterAssistantMessage(
                contact.id,
                text
            )

            sent += text
        }

        if (sent.isEmpty()) return null

        db.updateContact(
            contact.copy(
                lastProactiveAt = System.currentTimeMillis()
            )
        )

        return sent.joinToString("\n")
    }

    private suspend fun generateGuardedStage(
        contact: Contact,
        model: String,
        bundle: PromptBundle,
        maxTokens: Int,
        onStreaming: (String) -> Unit
    ): String? {
        var previewStarted = false
        var suspiciousPrefix = false

        val output = gateway.streamChat(
            provider = contact.provider,
            model = model,
            systemPrompt = bundle.system,
            userPrompt = bundle.user,
            maxTokens = maxTokens,
            onPartialText = partial@ { partial ->
                // Withhold the first few characters so very short repeats such as
                // “好累”“好困” can be rejected before the user sees them.
                if (!previewStarted) {
                    if (partial.length < 8) {
                        return@partial
                    }

                    suspiciousPrefix =
                        repetitionGuard.isSuspiciousPrefix(
                            contact.id,
                            partial.take(20)
                        )

                    if (!suspiciousPrefix) {
                        previewStarted = true
                        onStreaming(partial)
                    }
                } else {
                    onStreaming(partial)
                }
            }
        ).trim()

        if (output.isBlank()) {
            return null
        }

        val repeated =
            repetitionGuard.isRepetitive(
                contact.id,
                output
            )

        if (!repeated) {
            if (!previewStarted) {
                onStreaming(output)
            }
            return output
        }

        onStreaming("")

        // One regeneration attempt. If it is still repetitive,
        // silence is preferable to repeating the same line again.
        val replacement = gateway.chat(
            provider = contact.provider,
            model = model,
            systemPrompt =
                bundle.system +
                    """

                    【重复保护】
                    刚才生成的候选内容和你最近自己说过的话/主题过于相似。
                    这一次必须推进新信息、换自然角度，或简短回应后停住。
                    不要再次表达相同的“累、困、无聊、天气”等状态。
                    """.trimIndent(),
            userPrompt =
                bundle.user +
                    """

                    【最近已说内容，必须避开】
                    ${repetitionGuard.recentAssistantText(contact.id)}
                    """.trimIndent(),
            maxTokens = maxTokens
        ).trim()

        if (
            replacement.isBlank() ||
            repetitionGuard.isRepetitive(
                contact.id,
                replacement
            )
        ) {
            return null
        }

        onStreaming(replacement)
        return replacement
    }

    private fun isFastCasualMessage(text: String): Boolean {
        val clean = text.trim()
        if (clean.length > 42) return false

        val complexSignals = listOf(
            "详细", "分析", "解释", "为什么",
            "怎么做", "帮我", "代码", "方案",
            "比较", "总结", "原理", "论文",
            "计算", "证明"
        )

        return complexSignals.none {
            clean.contains(it)
        }
    }

    private fun naturalDelay(
        text: String,
        index: Int
    ): Long {
        val length = text.trim()
            .length
            .coerceAtLeast(1)

        val base = when {
            length <= 3 -> 280L
            length <= 10 -> 430L
            length <= 25 -> 700L
            length <= 60 -> 1000L
            else -> 1350L
        }

        return (
            base +
                stableJitter(
                    text.hashCode().toLong(),
                    320L
                ) +
                index * 220L
            ).coerceIn(240L, 2200L)
    }

    private fun stableJitter(
        seed: Long,
        range: Long
    ): Long {
        val positive = seed and 0x7fffffffL
        return positive % range.coerceAtLeast(1L)
    }

    private fun parseJson(raw: String): JsonObject? {
        val clean = raw.trim()
            .removePrefix("```json")
            .removePrefix("```JSON")
            .removePrefix("```")
            .removeSuffix("```")
            .trim()

        runCatching {
            JsonParser.parseString(clean).asJsonObject
        }.getOrNull()?.let {
            return it
        }

        val start = clean.indexOf('{')
        val end = clean.lastIndexOf('}')

        if (start >= 0 && end > start) {
            return runCatching {
                JsonParser.parseString(
                    clean.substring(
                        start,
                        end + 1
                    )
                ).asJsonObject
            }.getOrNull()
        }

        return null
    }

    private fun parseTime(value: String): Long? {
        if (
            value.isBlank() ||
            value.equals("null", true)
        ) {
            return null
        }

        return try {
            OffsetDateTime.parse(value)
                .toInstant()
                .toEpochMilli()
        } catch (_: DateTimeParseException) {
            null
        }
    }
}
