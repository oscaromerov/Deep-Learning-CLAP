# from experiments.text_to_audio import run_experiment as run_text_to_audio
# from experiments.audio_classification import run_experiment as run_audio_classification
def main():
    print("Select the experiment you want to run:")
    print("1. Text to Audio")
    print("2. Audio Classification")
    experiment = input("Enter the number of your choice: ")

    print("Select the dataset you want to use:")
    print("1. ESC50Dataset")
    print("2. GTZANDataset")
    print("3. UrbanSound8kDataset")
    dataset = input("Enter the number of your choice: ")

    if experiment == '1':
        if dataset == '1':
            run_text_to_audio('ESC50Dataset')
        elif dataset == '2':
            run_text_to_audio('GTZANDataset')
        elif dataset == '3':
            run_text_to_audio('UrbanSound8kDataset')
        else:
            print("Invalid dataset choice.")
    elif experiment == '2':
        if dataset == '1':
            run_audio_classification('ESC50Dataset')
        elif dataset == '2':
            run_audio_classification('GTZANDataset')
        elif dataset == '3':
            run_audio_classification('UrbanSound8kDataset')
        else:
            print("Invalid dataset choice.")
    else:
        print("Invalid experiment choice.")

if __name__ == "__main__":
    main()