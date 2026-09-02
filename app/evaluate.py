from app.agent import agent


TESTS = [
    {
        "question": "How can my company protect itself against foreign exchange risk?",
        "expected_tool": "search_products",
    },
    {
        "question": "Compare FX Swap and Range Forward.",
        "expected_tool": "compare_products",
    },
    {
        "question": "Does the documentation define target-market rules?",
        "expected_tool": "check_compliance",
    },
]


passed = 0

for test in TESTS:

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": test["question"],
                }
            ]
        }
    )

    called_tools = []

    for message in result["messages"]:

        if hasattr(message, "tool_calls"):

            for call in message.tool_calls:
                called_tools.append(call["name"])

    success = test["expected_tool"] in called_tools

    if success:
        passed += 1

    print("\nQUESTION:", test["question"])
    print("EXPECTED:", test["expected_tool"])
    print("CALLED:", called_tools)
    print("PASS:", success)


print("\n" + "=" * 60)
print(f"TOOL ROUTING: {passed}/{len(TESTS)}")
print("=" * 60)