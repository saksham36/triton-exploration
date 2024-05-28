import torch


def main():
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    
if __name__ is '__main__':
    main()