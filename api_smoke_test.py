from __future__ import annotations

import os
from dotenv import load_dotenv
from openai import OpenAI


def main() -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env")

    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model=model,
        input="Reply with exactly: OpenAI API is working."
    )

    print(response.output_text)


if __name__ == "__main__":
    main()
