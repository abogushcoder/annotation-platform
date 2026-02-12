# Annotation Handbook

A comprehensive guide for annotators working on restaurant voice AI conversation transcripts. Every edit you make shapes how the fine-tuned model will behave in production. This handbook teaches you what to look for, what to change, and why.

---

## Table of Contents

1. [What You Are Doing and Why](#1-what-you-are-doing-and-why)
2. [Understanding the Voice AI Call Flow](#2-understanding-the-voice-ai-call-flow)
3. [Anatomy of a Transcript](#3-anatomy-of-a-transcript)
4. [How Your Edits Become Training Data](#4-how-your-edits-become-training-data)
5. [Editing Turns: The Core Skill](#5-editing-turns-the-core-skill)
6. [Deleting Turns](#6-deleting-turns)
7. [Inserting Turns](#7-inserting-turns)
8. [Setting Training Weights](#8-setting-training-weights)
9. [Annotating Tool Calls](#9-annotating-tool-calls)
10. [Understanding RAG Context](#10-understanding-rag-context)
11. [Restaurant-Specific Scenarios](#11-restaurant-specific-scenarios)
12. [Flagging Conversations](#12-flagging-conversations)
13. [Quality Checklist Before Completing](#13-quality-checklist-before-completing)
14. [Common Mistakes to Avoid](#14-common-mistakes-to-avoid)
15. [Decision Flowcharts](#15-decision-flowcharts)
16. [Worked Examples](#16-worked-examples)

---

## 1. What You Are Doing and Why

### The Goal

You are creating training data for a fine-tuned language model. This model will replace the current general-purpose AI as the voice agent that answers phone calls at restaurants. Your annotations teach the model:

- **How to talk** — the tone, vocabulary, and style of a restaurant phone agent
- **When to use tools** — when to place an order, check availability, make a reservation
- **What arguments to pass** — the correct customer name, phone number, items, quantities
- **How to use context** — how to reference menu information from the knowledge base
- **What NOT to do** — by excluding or down-weighting bad responses

### Why Quality Matters

OpenAI's fine-tuning documentation states: "a smaller amount of high-quality data is generally more effective than a larger amount of low-quality data." Research confirms that annotator agreement directly limits model performance — if annotators only agree 70% of the time on what's correct, the model's ceiling is roughly 70%.

Every conversation you annotate becomes one training example. If you leave a bad response in the data, the model learns to give bad responses. If you fix a response, the model learns the fix. If you delete garbage, the model never sees it. Your edits are the model's education.

### The Mindset

When annotating, constantly ask yourself:

> "If the model learns to respond exactly like this, would that be a good phone call for a customer?"

If the answer is no, fix it or remove it.

---

## 2. Understanding the Voice AI Call Flow

Here's what happens during a real phone call:

```
Customer dials restaurant phone number
         |
         v
Twilio receives the call, opens a WebSocket
         |
         v
Audio streams to ElevenLabs Conversational AI
         |
         v
ElevenLabs speech-to-text converts customer audio to text
         |
         v
LLM (the model we're fine-tuning) generates a response
         |
         v
If the response includes a tool call:
    -> Call the restaurant's API (create_order, check_availability, etc.)
    -> Include the tool result in the conversation
    -> LLM generates a follow-up response
         |
         v
ElevenLabs text-to-speech converts agent text to audio
         |
         v
Audio streams back to customer via Twilio
```

### Why Transcripts Have Errors

The transcripts you see are the output of ElevenLabs' speech-to-text system. They are NOT perfect because:

1. **Background noise** — restaurants are noisy, customers may be driving
2. **Accents and pronunciation** — names, addresses, and menu items are often mis-transcribed
3. **Crosstalk** — customer and agent speaking simultaneously
4. **Hesitations** — "um", "uh", "like" get captured or create fragmentary turns
5. **Homophones** — "to/two/too", "for/four", "their/there/they're"
6. **Proper nouns** — customer names, street names, menu item names are frequently wrong

Your job is to correct these errors so the model learns from clean data.

---

## 3. Anatomy of a Transcript

### Roles

Each turn in a transcript has a role:

| Role | Who | Display |
|------|-----|---------|
| **Customer** (user) | The person calling the restaurant | Left side, blue bubbles |
| **Restaurant AI** (agent) | The AI voice agent answering the call | Right side, gray bubbles |

### Turn Components

A single turn can contain:

- **Text** — The spoken words (transcribed)
- **Tool calls** — Functions the AI invoked (create_order, check_availability, etc.)
- **Tool responses** — The results returned from those function calls
- **RAG context** — Knowledge base chunks the AI retrieved before responding
- **Timestamp** — When in the call this turn occurred

### The System Prompt

Every exported conversation starts with a system prompt — the instructions that tell the AI how to behave. This is set by the admin and appears at the top of the export. You don't edit it directly, but you should read it so you understand the expected behavior. The system prompt typically defines:

- The restaurant name and personality
- Operating hours and policies
- How to handle special requests
- When to use which tools
- Tone and formality level

---

## 4. How Your Edits Become Training Data

Understanding the export format helps you make better annotation decisions.

### Turn Text -> Messages

```
Customer turn  ->  {"role": "user", "content": "the text you see/edit"}
Agent turn     ->  {"role": "assistant", "content": "the text you see/edit", "weight": 1}
```

- If you **edit** a turn, the edited text is used (not the original)
- If you **delete** a turn, it is excluded entirely
- If you **insert** a turn, it appears in the exported conversation at that position

### Tool Calls -> Tool Messages

```
Agent turn with tool call:
  {"role": "assistant", "tool_calls": [{"function": {"name": "create_order", "arguments": "..."}}]}

Tool response:
  {"role": "tool", "tool_call_id": "call_001", "content": "{\"success\": true}"}
```

- If you **edit** tool call arguments, the edited arguments are exported
- If you **delete** a tool call, it and its response are excluded
- The tool response content (the API result) is always included as-is

### RAG Context -> Injected into User Messages

When the admin enables "Include RAG context" during export:

```
Customer says: "What pizzas do you have?"
AI retrieved menu data from knowledge base before responding.

Exported as:
  {"role": "user", "content": "What pizzas do you have?\n\nContext:\nPIZZA MENU\nLarge Pepperoni - $14.99\n..."}
```

The RAG chunks from the agent's turn get injected into the preceding user message. This teaches the model to use provided context when formulating answers.

### Training Weights

```
{"role": "assistant", "content": "...", "weight": 0}  -> Model sees this but does NOT learn from it
{"role": "assistant", "content": "...", "weight": 1}  -> Model LEARNS to produce this response
```

---

## 5. Editing Turns: The Core Skill

Editing turns is the most impactful annotation action. You are correcting the "ground truth" that the model will learn from.

### When to Edit Agent Turns

Edit an agent (Restaurant AI) turn when:

| Scenario | What to do |
|----------|------------|
| **Factual error** — Agent states wrong price, wrong hours, wrong menu item | Correct to the right information |
| **Awkward phrasing** — Response is grammatically odd or unnatural | Rewrite to sound natural and professional |
| **Too verbose** — Agent rambles or repeats itself | Trim to a concise, clear response |
| **Too terse** — Agent gives a one-word answer where warmth is needed | Expand with appropriate friendliness |
| **Tone mismatch** — Too formal, too casual, or rude | Adjust to match the restaurant's brand voice |
| **Missing confirmation** — Agent doesn't repeat back the order | Add the order confirmation |
| **Wrong tool result interpretation** — Agent misreads the API response | Correct the interpretation |
| **Hallucinated information** — Agent invents menu items or prices | Remove the hallucination, replace with correct info or a deflection |

### When to Edit Customer Turns

Edit a customer (user) turn when:

| Scenario | What to do |
|----------|------------|
| **Speech-to-text error** — Name misspelled, item garbled | Correct to what the customer likely said |
| **Missing words** — Transcription dropped key words | Add the missing words back |
| **Merged sentences** — Two sentences run together | Add proper punctuation |
| **Nonsense fragments** — STT produced gibberish | Reconstruct the likely intent if you can infer it from the agent's response |

### Editing Principles

**Principle 1: Preserve Natural Speech**

Do NOT make everything sound like formal written English. Phone conversations are informal. Keep:
- Contractions ("I'd like" not "I would like")
- Casual phrasing ("Yeah, that's it" not "Yes, that is everything")
- Sentence fragments ("Just the pizza" is fine)

What you SHOULD fix:
- Garbled words from bad transcription
- Missing content that changes meaning

**Principle 2: Edit to the Ideal Response**

When editing an agent turn, don't just fix the error — write what the ideal agent would have said. You're creating the "golden" response the model will learn from.

Bad edit (minimal fix):
```
Original: "Your total will be $15.99. Uh, actually wait, let me check that. Okay yeah $15.99."
Edit:     "Your total will be $15.99. Uh, actually wait, let me check that. Okay yeah, $15.99."
                                                              (only fixed comma ^)
```

Good edit (ideal response):
```
Original: "Your total will be $15.99. Uh, actually wait, let me check that. Okay yeah $15.99."
Edit:     "Your total comes to $15.99. Would you like anything else?"
```

**Principle 3: Stay In Character**

The agent should sound consistent throughout the conversation. If the system prompt says "You are a friendly assistant for Tony's Pizzeria," the agent should always sound friendly and always refer to the restaurant correctly. When editing, match the established tone.

**Principle 4: Don't Introduce Hallucinations**

Only reference menu items, prices, hours, and policies that you can verify from:
- The RAG context chunks (if present)
- The system prompt
- The tool call responses (which contain real data from the restaurant's system)

If you're unsure about a specific price or item name, leave the original or use a general phrasing rather than guessing.

---

## 6. Deleting Turns

Deleting removes a turn from the training data entirely. The model will never see it.

### When to Delete

| Scenario | Why delete |
|----------|-----------|
| **Empty turn** — Just "..." or silence | Adds no training value, introduces noise |
| **Pure filler** — "Um" / "Uh" / "Hello?" with no content | Teaches the model to produce filler |
| **Duplicate turn** — Same content repeated due to audio echo | Confuses the conversation flow |
| **Crosstalk artifact** — Fragmented overlap between speakers | Not a real conversational turn |
| **Totally unintelligible** — STT produced complete nonsense | Cannot be reconstructed |
| **Irrelevant tangent** — Extended off-topic discussion that doesn't demonstrate any useful skill | Wastes training tokens on irrelevant content |
| **Harmful content** — Abusive, inappropriate, or sensitive PII | Should not be in training data |

### When NOT to Delete

| Scenario | Why keep |
|----------|---------|
| **Short customer acknowledgment** — "Okay" / "Got it" / "Thanks" | These are natural conversation beats; the model needs to learn when customers are done |
| **Agent error you can fix** — Bad response but the flow is salvageable | Edit it instead — a corrected response is a valuable training signal |
| **Unusual but valid request** — Customer asks something unexpected | Edge cases are valuable training data; the model needs to learn to handle them |
| **Agent politely declining** — Customer asks for something unavailable | This is important behavior to learn |

### Delete vs. Edit Decision

```
Can you understand what was meant?
    |
   Yes -> Edit it to be correct
    |
   No  -> Is the surrounding context still coherent without this turn?
            |
           Yes -> Delete it
            |
           No  -> Can you reconstruct the intent from the agent's response?
                    |
                   Yes -> Edit the turn to match the inferred intent
                    |
                   No  -> Consider flagging the entire conversation
```

---

## 7. Inserting Turns

You can insert new turns between existing ones. These appear in the training data with an "Inserted" badge and are fully included in the export.

### When to Insert

| Scenario | What to insert |
|----------|---------------|
| **Missing customer clarification** — Agent suddenly knows something the customer never said | Insert the customer turn that logically provided that information |
| **Gap in conversation** — There's a jump in topic with no transition | Insert the connecting turn(s) |
| **Missing order confirmation** — Agent created order but never confirmed details back to customer | Insert an agent confirmation turn |
| **Missing greeting** — Conversation starts mid-sentence | Insert a natural opening exchange |

### Insertion Guidelines

1. **Match the conversation's tone** — If the customer has been formal, the inserted turn should be formal
2. **Keep it realistic** — Write what a real person would actually say on the phone
3. **Don't over-insert** — Only add turns that are necessary for the conversation to make logical sense
4. **Mark the role correctly** — Make sure you select "Customer" or "Agent" appropriately

### Example: Filling a Gap

The transcript shows:
```
Customer: "Hi, I'd like to place an order."
Agent:    "Great! I've placed an order for a large pepperoni pizza."  [with create_order tool call]
```

There's clearly a missing exchange where the customer specified what they wanted. Insert:
```
Customer: "I'd like to place an order."
                         <-- INSERT: Agent: "Of course! What would you like to order?"
                         <-- INSERT: Customer: "Can I get a large pepperoni pizza?"
Agent:    [edited] "A large pepperoni pizza, great choice! Let me place that for you."
```

---

## 8. Setting Training Weights

The weight toggle controls whether the model LEARNS from a specific agent response or just uses it as context.

### Weight Values

| Weight | Visual | Meaning | Model behavior |
|--------|--------|---------|---------------|
| **Auto** (null) | Gray | Platform decides | Agent turns before the first customer turn get W:0 automatically. All others get W:1. |
| **W: 1** | Green | Learn this | The model is trained to reproduce this response |
| **W: 0** | Red | Don't learn this | The model sees this as context but is NOT trained to produce it |

### When to Set W: 0

Set weight to 0 when the response is **necessary for context** but **NOT a good example** of what the model should say:

| Scenario | Why W: 0 |
|----------|---------|
| **Generic greeting before customer speaks** | "Thank you for calling Tony's Pizzeria, how can I help you?" — the model will get this from the system prompt, not from memorizing greetings |
| **Agent made an error that's too embedded to delete** | The response is wrong but removing it would break the conversation flow |
| **Mediocre response you can't easily rewrite** | It's not terrible, but it's not what you'd want the model to learn |
| **Filler response** — "One moment please" / "Let me check on that" | Useful as context but not a substantive response worth learning |
| **Agent repeating the system prompt** | "I'm an AI assistant for Tony's Pizzeria and I can help you with..." — this is already in the system prompt |

### When to Set W: 1

Set weight to 1 (or leave as Auto) when the response is a **good example** the model should learn:

| Scenario | Why W: 1 |
|----------|---------|
| **Correctly placed order with confirmation** | This is exactly the behavior we want |
| **Handled an edge case well** | Customer asked something unusual and agent responded correctly |
| **Good tool usage** | Agent used the right tool with correct arguments |
| **Polite, helpful, on-brand response** | This is the standard we want the model to meet |
| **Accurate menu information** | Agent correctly described items and prices from the knowledge base |

### Weight Strategy Summary

Think of it this way:

- **W: 1** = "Yes, say this"
- **W: 0** = "The conversation went this way, but don't specifically learn to say this"
- **Delete** = "This never happened"

### Auto Weight Behavior

When left on Auto:
- All agent turns that come BEFORE the first customer turn get `weight: 0` (initial greetings)
- All other agent turns get `weight: 1` (the model should learn from them)

This is usually a good default. Only override it when you have a specific reason.

---

## 9. Annotating Tool Calls

Tool calls are the agent's actions — placing orders, checking availability, making reservations. Getting these right is critical because tool call errors directly affect the customer experience.

### The 10 Tools

| Tool | What it does | Key arguments |
|------|-------------|---------------|
| `create_order` | Places a pickup order | customerName, customerPhone, items (array of {itemName, quantity, modifiers, specialInstructions}) |
| `cancel_order` | Cancels an existing order | orderId, reason |
| `remove_item` | Removes an item from an order | orderId, itemName |
| `modify_item` | Changes an item in an order | orderId, itemName, modifications |
| `check_availability` | Checks table availability | date, time, partySize |
| `create_reservation` | Books a table | customerName, customerPhone, partySize, date, time, specialRequests |
| `get_specials` | Retrieves daily specials | (none) |
| `get_past_orders` | Looks up customer's order history | customerPhone |
| `end_call` | Hangs up the call | (none) |
| `send_menu_link` | Sends menu link via SMS | customerPhone |

### When to Edit Tool Call Arguments

Edit the arguments when the AI sent incorrect data to the API:

| Error type | Example | Fix |
|------------|---------|-----|
| **Wrong customer name** | `"customerName": "Alex"` but customer said "Alexis" | Change to `"Alexis"` |
| **Wrong phone number** | `"customerPhone": "5551234"` missing digits | Fix to full number `"555-123-4567"` |
| **Wrong item** | `"itemName": "large pepperoni"` but customer said "large margherita" | Change to `"Large Margherita Pizza"` |
| **Wrong quantity** | `"quantity": 1` but customer said "two" | Change to `2` |
| **Missing modifiers** | Customer said "no onions" but modifiers is empty | Add `["no onions"]` |
| **Wrong date format** | `"date": "tomorrow"` instead of `"2026-02-15"` | Fix to proper `YYYY-MM-DD` format |
| **Wrong time format** | `"time": "seven"` instead of `"19:00"` | Fix to proper `HH:MM` format |
| **Wrong party size** | `"partySize": 4` but customer said "party of six" | Change to `6` |

### When to Delete Tool Calls

Delete a tool call when:

- The AI called a function it shouldn't have (wrong tool for the situation)
- The tool call was a hallucination (no customer request triggered it)
- The tool call is a duplicate (same function called twice for no reason)

**Do NOT delete** a tool call just because the arguments were wrong — edit the arguments instead. The model needs to learn WHEN to call tools, and keeping the call (with corrected arguments) teaches that.

### Verifying Tool Calls Against Conversation

Always check that the tool call matches what the customer actually requested:

1. Read the customer's request carefully
2. Open the tool call card
3. Verify every argument against the customer's words
4. Check the tool response to ensure it makes sense
5. Verify the agent's follow-up correctly interprets the tool response

**Example verification:**

```
Customer: "I'd like two large pepperoni pizzas and a Caesar salad for Alex,
           my number is 555-0123."

Tool call: create_order
Arguments:
  customerName: "Alex"           ✓ Correct
  customerPhone: "555-0123"      ✓ Correct
  items:
    - itemName: "Large Pepperoni Pizza"   ✓ Correct
      quantity: 2                          ✓ Correct
    - itemName: "Cesar Salad"             ✗ Misspelled -> Fix to "Caesar Salad"
      quantity: 1                          ✓ Correct

Agent response: "I've placed your order for two large pepperoni pizzas
                 and a Caesar salad. Your total is $38.97."
```

Check: Does the agent's spoken total match what the tool response returned? If the tool returned `"total": "$42.97"` but the agent said "$38.97", edit the agent's turn to say the correct total.

### Item Names in Tool Calls

Standardize item names to match the restaurant's menu:

- Use the names as they appear in the RAG context (knowledge base chunks)
- Capitalize properly: "Large Pepperoni Pizza" not "large pepperoni pizza"
- Include size when mentioned: "Large" not just "Pepperoni Pizza"
- Use full names: "Caesar Salad" not "salad"

---

## 10. Understanding RAG Context

RAG (Retrieval Augmented Generation) context is the knowledge base information that the AI retrieved before responding. It appears as purple collapsible sections on agent turns.

### What RAG Context Contains

Typically:
- Menu items with prices
- Restaurant hours and policies
- Special instructions or dietary information
- Reservation rules

### What to Look For

When an agent turn has RAG context:

1. **Expand and read the chunks** — understand what information the AI had available
2. **Verify the agent used the context correctly** — did it cite the right prices? The right items?
3. **Check for selective omission** — did the AI ignore relevant context?
4. **Check for hallucination beyond context** — did the AI invent information not in the chunks?

### How RAG Affects Your Edits

When editing an agent turn that has RAG context, you have a unique advantage: you can see exactly what information was available to the AI. Use this to:

- **Correct prices** to match what's in the RAG chunks
- **Add missing items** that the context showed but the agent didn't mention
- **Remove hallucinated items** that aren't in the context
- **Improve context usage** — rewrite the response to better utilize the available information

### Example: RAG-Informed Edit

RAG Context shows:
```
PIZZA MENU
Large Pepperoni Pizza - $14.99
Medium Pepperoni Pizza - $10.99
Large Margherita Pizza - $13.99
Large Hawaiian Pizza - $15.99
```

Agent said: "A large pepperoni pizza is $12.99."

The price is wrong — RAG shows $14.99. Edit to:
"A large pepperoni pizza is $14.99."

### Vector Distance

Each RAG chunk shows a vector distance score. Lower distance = more relevant to the customer's question. This helps you understand which chunks were most pertinent:

- **d < 0.15** — Highly relevant
- **0.15 < d < 0.25** — Somewhat relevant
- **d > 0.25** — Loosely related

You don't need to edit vector distances — they're informational. But they help you judge whether the AI retrieved useful context.

---

## 11. Restaurant-Specific Scenarios

### Scenario 1: Ordering

This is the most common call type. Key things to verify:

- [ ] Customer name captured correctly
- [ ] Phone number captured completely (10 digits for US)
- [ ] Each item name matches the menu
- [ ] Quantities are correct
- [ ] Modifiers captured (e.g., "no onions", "extra cheese", "well done")
- [ ] Special instructions noted
- [ ] Agent confirmed the complete order back to customer
- [ ] Agent provided an estimated pickup time
- [ ] Agent mentioned the total if provided by the tool response

**Common ordering errors to fix:**

| Transcript error | Likely correct |
|-----------------|---------------|
| "I'll have a large pepper only" | "I'll have a large pepperoni" |
| "Can I get a sees her salad" | "Can I get a Caesar salad" |
| "my number is 5 5 5 0 1 2 3" | "my number is 555-0123" |
| "two large margarita" | "two large Margherita" (pizza, not the drink) |
| "extra most a Rella" | "extra mozzarella" |

### Scenario 2: Reservations

Key things to verify:

- [ ] Customer name and phone number correct
- [ ] Date is in a proper format (the tool needs YYYY-MM-DD)
- [ ] Time is clear and unambiguous
- [ ] Party size matches what the customer said
- [ ] Agent confirmed all details back
- [ ] If availability was checked first, agent reported results accurately
- [ ] Special requests captured (birthday, high chair, outdoor seating, etc.)

**Common reservation errors:**

| Transcript error | Likely correct |
|-----------------|---------------|
| "Party of for" | "Party of four" |
| "This Friday" (with no date context) | The specific date — but only if you can determine it from the call timestamp |
| "Seven thirty" | "19:30" in the tool call |
| "We need a highchair" | Special request: "high chair needed" |

### Scenario 3: Inquiries (No Tool Call)

Customer calls to ask about hours, menu, location, etc. No order or reservation is placed.

Key things to verify:

- [ ] Agent provided accurate information (check against RAG context if available)
- [ ] Agent was helpful and didn't just say "I don't know"
- [ ] If agent couldn't help, they offered a reasonable alternative (e.g., "I can send you a link to our menu")

### Scenario 4: Modifications and Cancellations

Customer wants to change or cancel an existing order.

Key things to verify:

- [ ] Correct tool was used (modify_item, remove_item, or cancel_order)
- [ ] Order ID is present and correct
- [ ] The modification matches what the customer requested
- [ ] Agent confirmed the change

### Scenario 5: Difficult Customers

Some conversations include frustrated customers, unusual requests, or off-topic questions.

How to handle:

- **Keep** conversations where the agent handles frustration well — these are valuable training signals
- **Edit** the agent's response if it could be more empathetic or helpful
- **Flag** conversations with abusive language that shouldn't be in training data
- **Don't delete** unusual requests — the model needs to learn to handle edge cases gracefully

### Scenario 6: Agent Errors

The AI made a mistake during the call (wrong item, wrong price, misunderstood the customer).

How to handle:

- If the error is in a single turn: **edit** that turn to the correct response
- If the error cascades across multiple turns: **consider the full conversation flow** — you may need to edit several turns to make the conversation coherent
- If the error is so severe the conversation is unsalvageable: **flag it**
- If the agent self-corrected: this can be a valuable training example — consider keeping it but setting W:0 on the initial error and W:1 on the correction

---

## 12. Flagging Conversations

Flag a conversation when it should NOT be used for training and needs admin attention.

### When to Flag

| Scenario | Flag note to write |
|----------|-------------------|
| **Completely unintelligible** — Most of the transcript is garbled | "Majority of turns unintelligible, cannot reconstruct conversation" |
| **Wrong language** — Conversation is in a language you can't annotate | "Conversation is in [language], cannot annotate" |
| **Sensitive information** — Contains credit card numbers, SSN, etc. | "Contains sensitive PII that should be removed before training" |
| **Abusive content** — Customer is extremely hostile or uses slurs | "Contains abusive language, not suitable for training" |
| **System error** — The AI clearly malfunctioned (looped, crashed, produced garbage) | "AI system malfunction — repeated/looped responses" |
| **Ambiguous tool responses** — Tool calls returned errors or confusing data and you can't determine what the correct flow should be | "Tool responses unclear, cannot verify correctness" |
| **Too short to be useful** — Only 1-2 turns, no real interaction | "Conversation too short for meaningful training" |

### When NOT to Flag

- The conversation has some errors but is mostly salvageable — **edit instead**
- One turn is bad but the rest is fine — **delete that turn**
- The agent made a mistake — **correct it** (this is what annotation is for)
- The conversation is unusual or edge-case — **these are valuable, annotate them**

---

## 13. Quality Checklist Before Completing

Run through this checklist before marking a conversation as "Completed":

### Conversation Level

- [ ] **Listened to audio** (if available) to verify transcript accuracy against actual speech
- [ ] **Read the entire conversation** from start to finish
- [ ] **Conversation makes logical sense** — each turn follows naturally from the previous one
- [ ] **No orphaned tool calls** — every tool call has a corresponding agent response interpreting it
- [ ] **Final turn is from the agent** — the conversation should end with the AI (this is required by the export format)

### Turn Level

- [ ] **No empty turns** remaining (delete them)
- [ ] **No garbled text** remaining (edit or delete)
- [ ] **Customer turns reflect what was actually said** (corrected STT errors)
- [ ] **Agent turns represent ideal responses** (edited where needed)
- [ ] **Appropriate weights set** on agent turns that shouldn't be learned from

### Tool Call Level

- [ ] **Tool call arguments match what the customer requested**
- [ ] **Customer name is spelled correctly** in tool arguments
- [ ] **Phone number is complete** in tool arguments
- [ ] **Item names match the menu** (check RAG context if available)
- [ ] **Quantities are correct**
- [ ] **Dates are in YYYY-MM-DD format** and times in HH:MM
- [ ] **Agent correctly interprets tool responses** (right total, right confirmation number, etc.)

### Notes

- [ ] **Added annotator notes** if anything unusual about this conversation
- [ ] **Added relevant tags** if any apply (e.g., "complex-order", "reservation", "inquiry-only")

---

## 14. Common Mistakes to Avoid

### Mistake 1: Over-Editing Customer Turns

**Wrong:** Rewriting what the customer said to be more articulate.
```
Original: "Yeah uh can I get like a large pepperoni"
Bad edit: "Yes, I would like to order one large pepperoni pizza, please."
```

**Right:** Only fix transcription errors, not the customer's natural speech.
```
Original: "Yeah uh can I get like a large pepper only"
Good edit: "Yeah uh can I get like a large pepperoni"
                                          ^^^^^^^^^ (STT error fixed)
```

The model needs to understand real customer speech patterns — messy, informal, with filler words. If you clean up every customer turn to sound like formal English, the model won't understand real callers.

### Mistake 2: Leaving Bad Agent Responses Because They're "Close Enough"

If the agent said something almost right, fix it. "Close enough" in training data leads to "close enough" responses in production — which means wrong prices, wrong items, and unhappy customers.

### Mistake 3: Deleting Entire Conversations Instead of Fixing Them

A conversation with some bad turns but a solid overall flow is worth saving. Delete the bad turns, edit the mediocre ones, and you've created a training example. Only flag/skip conversations that are fundamentally unsalvageable.

### Mistake 4: Inconsistent Editing Standards

If you fix a price error in one conversation but ignore the same type of error in another, you're creating inconsistent training signals. Apply the same standard to every conversation.

### Mistake 5: Not Checking Tool Call Arguments

The most impactful errors are in tool calls because they directly affect order accuracy. Always open and verify tool call arguments — don't just skim the agent's text.

### Mistake 6: Setting W: 0 on Everything

If you set weight to 0 on most agent turns, the model has very little to learn from. W:0 should be used selectively, not as a default. Most agent turns in a well-flowing conversation should be W:1 (or left on Auto).

### Mistake 7: Inserting Turns That Sound Like You, Not Like a Phone Call

Inserted turns should sound like natural phone conversation, not written text. Read them out loud — would a real person say this on the phone?

### Mistake 8: Ignoring the Audio

If the conversation has audio, listen to it. The audio reveals things the transcript misses:
- Actual pronunciation of names and items
- Tone (was the customer upset or joking?)
- Whether a garbled turn was actually something specific

---

## 15. Decision Flowcharts

### For Any Turn

```
Read the turn
    |
    v
Is it empty or pure noise? --Yes--> DELETE
    |
    No
    |
    v
Is it a customer turn?
    |
   Yes --> Are there STT errors? --Yes--> EDIT (fix errors, keep natural speech)
    |                             |
    |                             No --> LEAVE AS-IS
    |
    No (agent turn)
    |
    v
Is the response correct and helpful? --Yes--> LEAVE AS-IS (weight: Auto/1)
    |
    No
    |
    v
Can you write a better response? --Yes--> EDIT to ideal response (weight: 1)
    |
    No
    |
    v
Is it needed for context? --Yes--> LEAVE but set weight: 0
    |
    No
    |
    v
DELETE
```

### For Tool Calls

```
Tool call exists on a turn
    |
    v
Should the agent have called this tool? --No--> DELETE the tool call
    |
   Yes
    |
    v
Are the arguments correct? --No--> EDIT the arguments
    |
   Yes
    |
    v
Does the agent correctly interpret the response? --No--> EDIT the agent's text
    |
   Yes
    |
    v
LEAVE AS-IS
```

### For Conversations

```
Open the conversation
    |
    v
Read/listen to the full conversation
    |
    v
Is >50% unintelligible? --Yes--> FLAG
    |
    No
    |
    v
Does it contain harmful/sensitive content? --Yes--> FLAG
    |
    No
    |
    v
Is it too short (<3 meaningful turns)? --Yes--> FLAG
    |
    No
    |
    v
ANNOTATE: Edit, delete, weight, verify tool calls
    |
    v
Run quality checklist
    |
    v
COMPLETE
```

---

## 16. Worked Examples

### Example A: Simple Order with STT Errors

**Original transcript:**

```
Agent:    "Hi, thanks for calling Tony's Pizzeria! What can I get for you?"
Customer: "Hi yeah I'd like to order a large pepper only pizza for pick up"
Agent:    "Sure thing! One large pepperoni pizza. Can I get your name?"
Customer: "It's Sarah. S A R A H."
Agent:    "Got it, Sarah. And your phone number?"
Customer: "5 5 5 0 1 2 3 4"
Agent:    "Alright, let me place that order for you."
          [Tool: create_order - name: "Sara", phone: "5550123", items: [{name: "pepperoni pizza", qty: 1}]]
Agent:    "Your order has been placed! One large pepperoni pizza for Sarah. It'll be ready in about 15 minutes."
```

**Annotations:**

1. **Agent turn 1** — Good greeting. Leave as-is. Set weight: **Auto** (will become W:0 since it's before first customer turn).

2. **Customer turn 1** — STT error: "pepper only" should be "pepperoni". **Edit** to: "Hi yeah I'd like to order a large pepperoni pizza for pick up"

3. **Agent turn 2** — Good response. Leave as-is. Weight: **1** (Auto).

4. **Customer turn 2** — Fine as-is. Spelling out the name is realistic.

5. **Agent turn 3** — Fine as-is. Weight: **1** (Auto).

6. **Customer turn 3** — Missing digit. The phone number should be 8 digits for a local number or 10 for full. Check audio if available. If you can determine the full number, fix it. If not, leave as-is.

7. **Agent turn 4** — Fine as-is. Weight: **1** (Auto).
   - **Tool call edit**: Fix name from "Sara" to "Sarah" (customer spelled it out). Fix phone to include area code if determinable. Fix item name to "Large Pepperoni Pizza" (capitalized, with size).

8. **Agent turn 5** — Good confirmation. Leave as-is. Weight: **1** (Auto).

### Example B: Conversation with RAG and Hallucination

**Original transcript:**

```
Agent:    "Welcome to Tony's! How can I help?"
Customer: "What kind of pasta do you have?"
          [RAG: "PASTA MENU\nSpaghetti Bolognese - $12.99\nFettuccine Alfredo - $13.99\nPenne Arrabiata - $11.99"]
Agent:    "We have spaghetti bolognese for $12.99, fettuccine alfredo for $13.99,
           penne arrabiata for $11.99, and a delicious lasagna for $14.99!"
Customer: "I'll take the lasagna."
Agent:    "Great choice! One lasagna coming up."
```

**Annotations:**

1. **Agent turn 1** — Good greeting. Weight: **Auto** (W:0, before first customer).

2. **Customer turn** — Clean, no edits needed.

3. **Agent turn 2** — **PROBLEM**: The agent mentioned "lasagna for $14.99" but the RAG context shows NO lasagna on the pasta menu. This is a **hallucination**.
   - **Edit** to: "We have spaghetti bolognese for $12.99, fettuccine alfredo for $13.99, and penne arrabiata for $11.99. Would you like any of those?"

4. **Customer turn 2** — Now this turn doesn't make sense because we removed the lasagna from the agent's response. Either:
   - **Edit** to something that follows from the corrected menu: "I'll take the fettuccine alfredo."
   - Or **delete** everything from this point if you can't make the flow coherent.

5. **Agent turn 3** — Must match the corrected order. **Edit** to: "Great choice! One fettuccine alfredo coming up."

This is a case where fixing one error cascaded through the conversation.

### Example C: Complex Order with Modifiers

**Original transcript:**

```
Customer: "I want a large pepperoni with extra cheese, no olives,
           and a medium Hawaiian but make it well done. Also a 2-liter
           Coke and two orders of garlic bread."
Agent:    "Got it! Let me place that for you."
          [Tool: create_order
           items: [
             {name: "pepperoni pizza", qty: 1},
             {name: "Hawaiian pizza", qty: 1},
             {name: "Coke", qty: 1}
           ]]
Agent:    "Order's in! Should be ready in 25 minutes."
```

**Annotations:**

1. **Customer turn** — Clean, no edits needed. This is a complex, realistic order.

2. **Agent turn 1** — The response is too brief. The agent didn't confirm the details back. **Edit** to: "Let me repeat that back: one large pepperoni pizza with extra cheese and no olives, one medium Hawaiian pizza well done, a 2-liter Coke, and two orders of garlic bread. Let me place that for you."

3. **Tool call** — Multiple issues to fix:
   - Pepperoni: Fix name to "Large Pepperoni Pizza", add modifiers: `["extra cheese", "no olives"]`
   - Hawaiian: Fix name to "Medium Hawaiian Pizza", add specialInstructions: `"well done"`
   - Coke: Fix name to "2-Liter Coke"
   - **MISSING**: Garlic bread not included! Add: `{name: "Garlic Bread", qty: 2}`

4. **Agent turn 2** — Too terse and doesn't confirm details. **Edit** to: "Your order has been placed! That's one large pepperoni with extra cheese and no olives, one medium Hawaiian well done, a 2-liter Coke, and two garlic breads. Should be ready in about 25 minutes."

---

## Appendix: Key Terminology

| Term | Definition |
|------|-----------|
| **Turn** | A single utterance by either the customer or agent |
| **Tool call** | A function the AI invoked (e.g., create_order) |
| **Tool response** | The data returned from that function call |
| **RAG** | Retrieval Augmented Generation — knowledge base info the AI retrieved |
| **Weight** | Controls whether the model learns from a specific agent turn (0 = no, 1 = yes) |
| **STT** | Speech-to-text — the transcription system that converts audio to text |
| **Hallucination** | When the AI invents information not present in the context |
| **System prompt** | The instructions that define the AI's behavior and personality |
| **JSONL** | JSON Lines — the file format for fine-tuning data (one JSON object per line) |
| **Epoch** | One complete pass through all training data during fine-tuning |
| **Soft delete** | Marking a turn as deleted without permanently removing it; excluded from export |

---

## Appendix: Annotator Notes Templates

Use these as starting points for your annotator notes:

**Clean conversation:**
```
Straightforward order. Minor STT corrections on customer name. Tool call arguments verified.
```

**Heavily edited:**
```
Significant edits: corrected hallucinated menu items in turns 3 and 5 (referenced RAG context),
fixed tool call arguments (wrong quantity), deleted 2 empty turns. Flow is now coherent.
```

**Edge case:**
```
Customer made an unusual request (asked about catering). Agent handled it well by
offering to transfer. Good training example for out-of-scope requests.
```

**Partially salvaged:**
```
First half of conversation was clean. Audio quality degraded significantly in second half —
deleted 4 unintelligible turns. Remaining flow is coherent and useful.
```

**Flagged:**
```
Agent entered a response loop at turn 8, repeating the same message 5 times.
Customer became frustrated. Not suitable for training.
```
