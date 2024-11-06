"""Python script for evaluating SLM output via GPT-4."""

import os

import openai

import config


def main():
    """Driver for script."""
    openai.api_key = os.environ["OPENAI_API_KEY"]
    client = openai.OpenAI(
        organization="org-k4X6o1QPuBpEC8GJahNSdpo4",
        project="proj_DUVloPcPSQnFty2IFRGocLkq",
    )

    response = client.chat.completions.create(
        model = "gpt-4o",
        messages = [
            {
                "role": "user",
                "content": [{ "type": "text", "text": "Tell me a cool fact!"}]
            }
        ]
    )

    print(response.choices[0].message)

    return 0


if __name__ == "__main__":
    main()
