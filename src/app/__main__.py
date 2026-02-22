from app.lightning_manager import LightningManager


def main():
    lightning_manager = LightningManager()
    lightning_manager.start_training()


if __name__ == "__main__":
    main()
