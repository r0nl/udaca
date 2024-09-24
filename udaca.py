import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, cohen_kappa_score

# Load datasets
def load_data():
    paviaU_data = loadmat('./sample_data/paviaU.mat')['ori_data']
    paviaU_gt = loadmat('./sample_data/paviaU_7gt.mat')['map']
    paviaC_data = loadmat('./sample_data/paviaC.mat')['ori_data']
    paviaC_gt = loadmat('./sample_data/paviaC_7gt.mat')['map']
    return paviaU_data, paviaU_gt, paviaC_data, paviaC_gt

# Feature Compaction Network
class FeatureCompactionNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeatureCompactionNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x

# Classifier
class Classifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

# Adversarial Discriminator for Domain-Level Alignment
class DomainDiscriminator(nn.Module):
    def __init__(self, feature_dim):
        super(DomainDiscriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

# Style-Perceive Alignment (SPA) using Gram matrix in mini-batches
def calculate_gram_matrix(x):
    """Compute Gram matrix for style alignment."""
    batch_size, num_features = x.size()
    gram_matrix = torch.mm(x, x.t())
    return gram_matrix / (batch_size * num_features)

def style_perceive_loss(source_features, target_features, batch_size=5000):
    """Style alignment using Gram matrices calculated in mini-batches."""
    num_samples_source = source_features.size(0)
    num_samples_target = target_features.size(0)
    loss = 0.0
    for i in range(0, min(num_samples_source, num_samples_target), batch_size):
        source_batch = source_features[i:i + batch_size]
        target_batch = target_features[i:i + batch_size]
        
        # Ensure both batches have the same size
        if source_batch.size(0) == target_batch.size(0):
            gram_source = calculate_gram_matrix(source_batch)
            gram_target = calculate_gram_matrix(target_batch)
            loss += torch.mean((gram_source - gram_target) ** 2)
    
    num_batches = min(num_samples_source, num_samples_target) // batch_size
    return loss / num_batches

# Loss functions
def classification_loss(preds, labels):
    return F.cross_entropy(preds, labels)

def entropy_loss(preds):
    return -torch.mean(torch.sum(preds * torch.log(preds + 1e-6), dim=1))

def domain_discriminator_loss(preds, domain_labels):
    return F.binary_cross_entropy(preds, domain_labels)

# Train the UDACA model
def train_model(paviaU_data, paviaU_gt, paviaC_data, paviaC_gt, source='paviaU', num_epochs=1000, batch_size=50):
    # Choose source and target datasets based on the input
    if source == 'paviaU':
        source_data, source_labels = paviaU_data, paviaU_gt
        target_data = paviaC_data
    else:
        source_data, source_labels = paviaC_data, paviaC_gt
        target_data = paviaU_data

    # Reshape data: Pixel-wise classification [Height, Width, Bands] -> [Pixels, Bands]
    source_data = torch.tensor(source_data, dtype=torch.float32).reshape(-1, 102)  # Flatten into [Pixels, Bands]
    target_data = torch.tensor(target_data, dtype=torch.float32).reshape(-1, 102)  # Flatten into [Pixels, Bands]
    source_labels = torch.tensor(source_labels, dtype=torch.long).reshape(-1)  # Flatten into [Pixels]

    # Initialize networks
    feature_net = FeatureCompactionNet(input_dim=102, output_dim=256)
    classifier = Classifier(feature_dim=256, num_classes=9)
    domain_discriminator = DomainDiscriminator(feature_dim=256)

    # Optimizers
    optimizer_feature = optim.Adam(feature_net.parameters(), lr=1e-4)
    optimizer_classifier = optim.Adam(classifier.parameters(), lr=1e-4)
    optimizer_domain = optim.Adam(domain_discriminator.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(num_epochs):
        # Feature extraction
        source_features = feature_net(source_data)
        target_features = feature_net(target_data)

        # Classifier loss on source domain (CWA)
        source_preds = classifier(source_features)
        loss_cls = classification_loss(source_preds, source_labels)

        # Entropy loss for target domain (unsupervised)
        target_preds = classifier(target_features)
        loss_ent = entropy_loss(F.softmax(target_preds, dim=1))

        # Domain Discriminator loss for domain-level alignment
        domain_preds_source = domain_discriminator(source_features)
        domain_preds_target = domain_discriminator(target_features)
        domain_labels_source = torch.ones(source_features.size(0), 1)  # Domain label 1 for source
        domain_labels_target = torch.zeros(target_features.size(0), 1)  # Domain label 0 for target
        loss_domain = domain_discriminator_loss(domain_preds_source, domain_labels_source) + \
                      domain_discriminator_loss(domain_preds_target, domain_labels_target)

        # Style-Perceive Alignment (SPA) loss
        loss_spa = style_perceive_loss(source_features, target_features)

        # Total loss
        total_loss = loss_cls + loss_ent + loss_domain + loss_spa

        # Backpropagation and optimization
        optimizer_feature.zero_grad()
        optimizer_classifier.zero_grad()
        optimizer_domain.zero_grad()
        total_loss.backward()
        optimizer_feature.step()
        optimizer_classifier.step()
        optimizer_domain.step()

        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {total_loss.item()}")

    print("Training Complete.")
    return feature_net, classifier

# Evaluation metrics
def evaluate_model(feature_net, classifier, data, labels):
    features = feature_net(data)
    preds = classifier(features)
    preds = torch.argmax(preds, dim=1)
    accuracy = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
    return accuracy, preds

# Overall Accuracy (OA), Average Accuracy (AA), Kappa Coefficient (Kappa)
def compute_metrics(predictions, labels):
    oa = accuracy_score(labels, predictions)
    aa = np.mean([accuracy_score(labels[labels == i], predictions[labels == i]) for i in np.unique(labels)])
    kappa = cohen_kappa_score(predictions, labels)
    return oa, aa, kappa

# Main
if __name__ == '__main__':
    paviaU_data, paviaU_gt, paviaC_data, paviaC_gt = load_data()

    # Train with PaviaU as source and PaviaC as target
    print("Training with PaviaU as source and PaviaC as target:")
    feature_net, classifier = train_model(paviaU_data, paviaU_gt, paviaC_data, paviaC_gt, source='paviaU')

    # Evaluate on target domain (PaviaC)
    target_data = torch.tensor(paviaC_data, dtype=torch.float32).reshape(-1, 102)  # [Pixels, Bands]
    target_labels = torch.tensor(paviaC_gt, dtype=torch.long).reshape(-1)  # Flattened [Pixels]
    
    target_accuracy, target_preds = evaluate_model(feature_net, classifier, target_data, target_labels)

    # Compute OA, AA, Kappa for target domain
    oa, aa, kappa = compute_metrics(target_preds.cpu().numpy(), target_labels.cpu().numpy())

    print(f"Target Domain - Overall Accuracy (OA): {oa}")
    print(f"Target Domain - Average Accuracy (AA): {aa}")
    print(f"Target Domain - Kappa Coefficient: {kappa}")
    
    # Train with PaviaC as source and PaviaU as target
    # print("\nTraining with PaviaC as source and PaviaU as target:")
    # feature_net, classifier = train_model(paviaU_data, paviaU_gt, paviaC_data, paviaC_gt, source='paviaC')