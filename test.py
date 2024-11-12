"""Python script for testing functionalities."""

from temp import construct_prompt


def main():
    """Driver for script."""
    user_prompt_1, user_prompt_2 = construct_prompt("Hello, world! Hello, world! Hello, world! Hello, world! Hello, world! Hello, world!")
    print(user_prompt_1, "\n\n\n\n\n\n", user_prompt_2)
    
    return 0


if __name__ == "__main__":
    main()
