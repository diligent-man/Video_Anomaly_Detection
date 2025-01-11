"""
Ref:
    1/ https://intellabs.github.io/distiller/knowledge_distillation.html#references
    2/ https://pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html
    3/ KLDiv: https://www.youtube.com/watch?app=desktop&v=Pwgpl9mKars
"""
import copy
import torch
import torchvision
from torchvision.transforms import v2, Compose


class TeacherModel(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(TeacherModel, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class StudentModel(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(StudentModel, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(16, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(1024, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def normal_train(model, train_loader, epochs, learning_rate, device):
    print("Starting normal train")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
        # with torch.amp.autocast("cuda", torch.float16):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")
    print()
    print()


def train_knowledge_distillation(teacher, student, train_loader,
                                 epochs, learning_rate,
                                 T, hard_loss_weight,
                                 device
                                 ):
    """In this training, we only train student model"""
    print("Starting knowledge distillation train")
    ce_loss = torch.nn.CrossEntropyLoss()
    kd_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)

    teacher.eval()  # Teacher set to evaluation mode and act as feature extractor
    student.train()  # Student to train mode

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
        # with torch.amp.autocast("cuda", torch.float16):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
            with torch.no_grad():
                teacher_logits = teacher(inputs)

            # Forward pass with the student model
            student_logits = student(inputs)

        # Calculate distillation loss as described in "Distilling the knowledge in a neural network"
        # input: soft_pred from student model
        # target: soft labels from teacher model
        # KLDivergence loss used to make soft pred from student more close to teacher soft pred
        soft_loss = kd_loss(
            torch.nn.functional.log_softmax(student_logits / T, -1),
            torch.nn.functional.log_softmax(teacher_logits / T, -1)
        ) * (T ** 2)

        # Calculate the true label loss
        hard_loss = ce_loss(student_logits, labels)

        # Weighted sum of the two losses
        loss = (1 - hard_loss_weight) * soft_loss + hard_loss_weight * hard_loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")
    print()
    print()


def test(model, test_loader, device):
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
        # with torch.amp.autocast("cuda", torch.float16):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test {model.__class__.__name__}: {accuracy:.2f}%")
    return None


def main() -> None:
    device = "cpu"
    torch.manual_seed(42)

    transforms_cifar = Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Loading the CIFAR-10 dataset:
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms_cifar)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms_cifar)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4096, shuffle=True, num_workers=32,
                                               persistent_workers=True, prefetch_factor=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4096, shuffle=False, num_workers=32,
                                              persistent_workers=True, prefetch_factor=4)
    print(f"Train dataloader: {len(train_loader)}, Test dataloader: {len(test_loader)}\n")

    # Train teacher model
    teacher_model = TeacherModel().to(device)
    student_model_1 = StudentModel().to(device)
    student_model_2 = copy.deepcopy(student_model_1)

    teacher_model.compile()
    student_model_1.compile()
    student_model_2.compile()

    total_params_deep = "{:,}".format(sum(p.numel() for p in teacher_model.parameters()))
    total_params_light = "{:,}".format(sum(p.numel() for p in student_model_1.parameters()))
    print(f"Teacher paras: {total_params_deep}")
    print(f"Student paras: {total_params_light}")
    print()

    print("Norm of 1st layer of student 1:", torch.norm(student_model_1.features[0].weight).item())
    print("Norm of 1st layer of student 2:", torch.norm(student_model_2.features[0].weight).item())
    print()

    normal_train(student_model_1, train_loader, 20, 1e-2, device)
    train_knowledge_distillation(
        teacher_model, student_model_2, train_loader,
        20, 1e-3,
        2, .3, device
    )

    test(student_model_1, test_loader, device)  # Latest test: Acc=62.93%
    test(student_model_2, test_loader, device)  # Latest test: Acc=70.42%
    return None


if __name__ == '__main__':
    main()
