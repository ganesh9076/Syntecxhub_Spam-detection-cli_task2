from src.train import train_models
from src.evaluate import evaluate_model
from src.visualize import (
    plot_confusion_matrix,
    plot_model_comparison,
    generate_wordclouds
)
from src.save_pipeline import save_pipeline
from src.predict import predict_message


def menu():

    while True:

        print("\n" + "=" * 40)
        print("ADVANCED SPAM DETECTION SYSTEM")
        print("=" * 40)

        print("1. Train Models")
        print("2. Evaluate Model")
        print("3. Generate Visualizations")
        print("4. Save Pipeline")
        print("5. Predict Message")
        print("6. Exit")

        choice = input(
            "\nEnter Choice: "
        )

        if choice == "1":
            train_models()
            print("\nTraining Completed")

        elif choice == "2":
            metrics, _ = evaluate_model()

            for key, value in metrics.items():
                print(
                    f"{key}: {value:.4f}"
                )

        elif choice == "3":
            plot_confusion_matrix()
            plot_model_comparison()
            generate_wordclouds()

            print(
                "\nVisualizations Generated"
            )
            
        elif choice == "4":
            save_pipeline()

            print(
                "\nPipeline Saved"
            )

        elif choice == "5":
            predict_message()

        elif choice == "6":
            break

        else:
            print(
                "\nInvalid Choice"
            )


if __name__ == "__main__":
    menu()