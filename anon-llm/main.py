import torch

def main():
    """
    Main entry point for project.
    """
    print(torch.__version__)
    print(torch.backends.mps.is_available())
    print("Anon is coming.")

if __name__ == "__main__":
    main()