# Script 2
class DoggerlandN2NDataset(Dataset):
    
    # Constructor that receives both noisy images, patch size, and the number of patches per epoch
    def __init__(self, noisy_1, noisy_2, patch_size=64, num_patches=1000):
        # Store patch size and quantity
        self.patch_size = patch_size
        self.num_patches = num_patches
        # Store the first noisy image and the second noisy image (same scene, different noise instance)
        self.n1 = noisy_1
        self.n2 = noisy_2
        # Retrieve and store height (h) and width (w) of the original matrix
        self.h, self.w = noisy_1.shape
    
    def __len__(self):
        return self.num_patches

    # Multi-indexing (idx)
    def __getitem__(self, idx):
        # Establish vertical and horizontal coordinates within bounds
        y = np.random.randint(0, self.h - self.patch_size)
        x = np.random.randint(0, self.w - self.patch_size)
        
        # Crop 64x64 pixel patches from both noisy sources
        patch_n1 = self.n1[y:y+self.patch_size, x:x+self.patch_size]
        patch_n2 = self.n2[y:y+self.patch_size, x:x+self.patch_size]
        
        # Output integrates the color channel dimension (Channel, Height, Width)
        return torch.tensor(patch_n1).unsqueeze(0), torch.tensor(patch_n2).unsqueeze(0)
