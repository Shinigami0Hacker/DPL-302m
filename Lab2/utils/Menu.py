class Menu:
    def __init__(self, content: dict) -> None:
        assert(isinstance(content, dict), TypeError("Content much be the list"))
        self.content = content

    def print(self) -> None:
        print(f"{'Menu':-^60}")
        for id, option in enumerate(self.content):
            print(f"{id + 1}: {option}")
        print("-" * 60)

    def validate(self, choice):
        if len(self.content) < choice or choice < 0:
            return False
        return True

    def updateContent(self, content: dict) -> None:
        assert(isinstance(content, list), TypeError("Content much be the list"))
        self.content = content
