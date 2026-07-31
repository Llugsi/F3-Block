# [Script 4] Dual-axis training loop (Self-Supervised MSE vs Joint Real PSNR)
class DoggerlandN2NDataset(Dataset):
    def __init__(self, noisy_1, noisy_2, clean=None, patch_size=64, num_patches=1000):
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.n1 = noisy_1
        self.n2 = noisy_2
        self.clean = clean  # Optional for Train, mandatory for Validation
        self.h, self.w = noisy_1.shape
    
    def __len__(self):
        return self.num_patches

    def __getitem__(self, idx):
        y = np.random.randint(0, self.h - self.patch_size)
        x = np.random.randint(0, self.w - self.patch_size)
        
        patch_n1 = self.n1[y:y+self.patch_size, x:x+self.patch_size]
        patch_n2 = self.n2[y:y+self.patch_size, x:x+self.patch_size]
        
        if self.clean is not None:
            patch_clean = self.clean[y:y+self.patch_size, x:x+self.patch_size]
            return (torch.tensor(patch_n1).unsqueeze(0), 
                    torch.tensor(patch_n2).unsqueeze(0), 
                    torch.tensor(patch_clean).unsqueeze(0))
        
        return torch.tensor(patch_n1).unsqueeze(0), torch.tensor(patch_n2).unsqueeze(0)


# =====================================================================
# AUXILIARY FUNCTION: Scientific PSNR Computation
# =====================================================================
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    # Since seismic data is standardized (mean 0, std 1), 
    # amplitude reference is set to 1.0 for standardized reflectors.
    max_pixel = 1.0  
    return 20 * np.log10(max_pixel / np.sqrt(mse))


# =====================================================================
# 1. DATA GENERATION (INDEPENDENT TRAIN AND VAL BLOCKS)
# =====================================================================
num_traces, num_samples = 256, 512
t = np.linspace(0, 10, num_samples)
x = np.arange(num_traces)

# Global Training Data Block
n1_train, n2_train, clean_train = seismic_data_generation(
    t, x, peak_1=1.5, peak_2=1.2, peak_3=0.9, axis=4.5, height=0.5, traces=128, m=0.005, b=7.0
)

# Global Validation Data Block (Geometrical modifications to prevent Overfitting)
n1_val, n2_val, clean_val = seismic_data_generation(
    t, x, peak_1=1.4, peak_2=1.1, peak_3=0.8, axis=5.0, height=0.7, traces=128, m=-0.003, b=6.5
)

# Instantiating Datasets and Loaders
train_dataset = DoggerlandN2NDataset(n1_train, n2_train, clean=None, patch_size=64, num_patches=1200)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=False)

val_dataset = DoggerlandN2NDataset(n1_val, n2_val, clean=clean_val, patch_size=64, num_patches=300)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=False)

# =====================================================================
# 2. ENVIRONMENT & LEARNING ENVIRONMENT SETUP
# =====================================================================
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TNNLS_BlindSpotNet().to(device)  

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 200  
train_loss_history = []
val_loss_history = []
val_psnr_history = []  

print(f"Starting Self-Supervised Noise2Noise training loop on {device}...")

# =====================================================================
# 3. TRAINING AND VALIDATION EXECUTION LOOP
# =====================================================================
for epoch in range(epochs):
    # --- TRAINING PHASE ---
    model.train()
    running_train_loss = 0.0
    for batch_n1, batch_n2 in train_loader:
        batch_n1, batch_n2 = batch_n1.to(device), batch_n2.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_n1)
        loss = criterion(outputs, batch_n2)
        
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()
        
    epoch_train_loss = running_train_loss / len(train_loader)
    train_loss_history.append(epoch_train_loss)
    
    # --- VALIDATION PHASE WITH JOINT PSNR EVALUATION ---
    model.eval()
    running_val_loss = 0.0
    running_val_psnr = 0.0
    
    with torch.no_grad():
        for batch_n1_v, batch_n2_v, batch_clean_v in val_loader:  
            batch_n1_v, batch_n2_v = batch_n1_v.to(device), batch_n2_v.to(device)
            
            outputs_v = model(batch_n1_v)
            val_loss = criterion(outputs_v, batch_n2_v)
            running_val_loss += val_loss.item()
            
            out_np = outputs_v.cpu().numpy()
            clean_np = batch_clean_v.numpy()
            running_val_psnr += calculate_psnr(clean_np, out_np)
            
    epoch_val_loss = running_val_loss / len(val_loader)
    epoch_val_psnr = running_val_psnr / len(val_loader)
    
    val_loss_history.append(epoch_val_loss)
    val_psnr_history.append(epoch_val_psnr)
    
    print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f} | Real Val PSNR: {epoch_val_psnr:.2f} dB")

print("Training and validation workflows successfully completed.")

# =====================================================================
# 4. SCIENTIFIC VISUALIZATION CODES (DUAL AXIS LOSS & PSNR TRENDS)
# =====================================================================
fig, ax1 = plt.subplots(figsize=(10, 5))

color_train = 'crimson'
color_val = 'dodgerblue'
ax1.set_xlabel("Training Epochs", fontsize=11)
ax1.set_ylabel("Loss Function (MSE)", fontsize=11, color='black')
line1 = ax1.plot(range(1, epochs + 1), train_loss_history, color=color_train, linewidth=2, label='Training Loss')
line2 = ax1.plot(range(1, epochs + 1), val_loss_history, color=color_val, linewidth=2, linestyle='--', label='Validation Loss')
ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(True, linestyle='--', alpha=0.5)

ax2 = ax1.twinx()  
color_psnr = 'forestgreen'
ax2.set_ylabel("Real Validation PSNR (dB)", fontsize=11, color=color_psnr)
line3 = ax2.plot(range(1, epochs + 1), val_psnr_history, color=color_psnr, linewidth=2, linestyle='-.', label='Real PSNR (vs Ground Truth)')
ax2.tick_params(axis='y', labelcolor=color_psnr)

lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right')

plt.title("Learning Curves and Seismic Reconstruction Trends", fontsize=12, fontweight='bold')
fig.tight_layout()
plt.show()
