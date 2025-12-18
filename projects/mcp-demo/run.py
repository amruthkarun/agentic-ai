from agent import Agent
import json

if __name__ == "__main__":
    agent = Agent()

    while True:
        q = input("\nAsk something (or 'exit'): ")
        if q.lower() == "exit":
            break

        out = agent.run(q)
        print(json.dumps(out, indent=2))
