import matplotlib.pyplot as plt

# BERT Training Data
bert_epochs = list(range(1, 26))
bert_train_loss = [
    1.7676, 1.6466, 1.5835, 1.5393, 1.5078, 1.4791, 1.4571, 1.4355,
    1.4219, 1.4101, 1.3939, 1.3802, 1.3691, 1.3596, 1.3455, 1.3389,
    1.3331, 1.3279, 1.3175, 1.3083, 1.3052, 1.2989, 1.2966, 1.2959, 1.2854
]
bert_valid_loss = [
    1.6664, 1.5639, 1.5072, 1.4718, 1.4413, 1.4178, 1.4017, 1.3833,
    1.3691, 1.3561, 1.3443, 1.3320, 1.3221, 1.3152, 1.3022, 1.2963,
    1.2917, 1.2862, 1.2785, 1.2721, 1.2675, 1.2610, 1.2586, 1.2551, 1.2552
]
bert_valid_acc = [
    0.6572, 0.6728, 0.6817, 0.6878, 0.6920, 0.6962, 0.6984, 0.7018,
    0.7041, 0.7062, 0.7084, 0.7102, 0.7119, 0.7130, 0.7150, 0.7160,
    0.7166, 0.7176, 0.7193, 0.7201, 0.7209, 0.7222, 0.7226, 0.7234, 0.7232
]

# DistilRoBERTa Training Data
distilroberta_epochs = bert_epochs
distilroberta_train_loss = [
    1.8516, 1.7505, 1.6889, 1.6438, 1.6115, 1.5891, 1.5691, 1.5577,
    1.5393, 1.5272, 1.5095, 1.4964, 1.4917, 1.4827, 1.4708, 1.4621,
    1.4527, 1.4481, 1.4353, 1.4374, 1.4302, 1.4209, 1.4145, 1.4117, 1.4104
]
distilroberta_valid_loss = [
    1.7216, 1.6323, 1.5803, 1.5401, 1.5150, 1.4958, 1.4787, 1.4625,
    1.4501, 1.4386, 1.4288, 1.4162, 1.4069, 1.4012, 1.3917, 1.3853,
    1.3766, 1.3723, 1.3642, 1.3635, 1.3534, 1.3506, 1.3464, 1.3451, 1.3444
]
distilroberta_valid_acc = [
    0.6453, 0.6587, 0.6675, 0.6739, 0.6781, 0.6812, 0.6840, 0.6864,
    0.6884, 0.6908, 0.6923, 0.6946, 0.6960, 0.6967, 0.6985, 0.6995,
    0.7012, 0.7017, 0.7033, 0.7035, 0.7050, 0.7056, 0.7063, 0.7066, 0.7066
]

# RoBERTa Training Data
roberta_epochs = bert_epochs
roberta_train_loss = [
    1.6348, 1.5497, 1.5088, 1.4835, 1.4355, 1.4159, 1.3914, 1.3813,
    1.3632, 1.3479, 1.3295, 1.3176, 1.3099, 1.3009, 1.2876, 1.2768,
    1.2674, 1.2615, 1.2482, 1.2476, 1.2391, 1.2288, 1.2219, 1.2192, 1.2162
]
roberta_valid_loss = [
    1.5317, 1.4585, 1.4545, 1.4191, 1.3693, 1.3505, 1.3294, 1.3144,
    1.3033, 1.2923, 1.2836, 1.2700, 1.2598, 1.2529, 1.2422, 1.2371,
    1.2271, 1.2226, 1.2135, 1.2117, 1.2007, 1.1973, 1.1929, 1.1913, 1.1894
]
roberta_valid_acc = [
    0.6787, 0.6892, 0.6914, 0.6940, 0.7043, 0.7071, 0.7107, 0.7126,
    0.7149, 0.7170, 0.7183, 0.7209, 0.7223, 0.7234, 0.7253, 0.7264,
    0.7278, 0.7288, 0.7302, 0.7307, 0.7323, 0.7333, 0.7341, 0.7346, 0.7344
]

# Visualization
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.plot(bert_epochs, bert_train_loss, 'o-', label='BERT')
plt.plot(distilroberta_epochs, distilroberta_train_loss, 's-', label='DistilRoBERTa')  # Square markers
plt.plot(roberta_epochs, roberta_train_loss, '^-', label='RoBERTa')  # Triangle markers
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Validation Loss with Different Markers
plt.subplot(1, 3, 2)
plt.plot(bert_epochs, bert_valid_loss, 'o-', label='BERT')
plt.plot(distilroberta_epochs, distilroberta_valid_loss, 's-', label='DistilRoBERTa')  # Square markers
plt.plot(roberta_epochs, roberta_valid_loss, '^-', label='RoBERTa')  # Triangle markers
plt.title('Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Validation Accuracy with Different Markers
plt.subplot(1, 3, 3)
plt.plot(bert_epochs, bert_valid_acc, 'o-', label='BERT')
plt.plot(distilroberta_epochs, distilroberta_valid_acc, 's-', label='DistilRoBERTa')  # Square markers
plt.plot(roberta_epochs, roberta_valid_acc, '^-', label='RoBERTa')  # Triangle markers
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()

plt.tight_layout()
plt.savefig('pretraining_process.png', dpi=300)
plt.show()
