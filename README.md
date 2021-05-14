# Alaska2_image_steganalysis
---
Detect secret data hidden within digital images

Steganalysis : Cover image에 secret message를 숨긴 Stego image를 판별해내는 것

사용된 Steganalysis 종류 : [JMiPOD](), [JUNIWARD](), [UERD]()

## Data 설명
This dataset contains a large number of unaltered images, called the "Cover" image, as well as corresponding examples in which information has been hidden using one of three steganography algorithms (JMiPOD, JUNIWARD, UERD). The goal of the competition is to determine which of the images in the test set (Test/) have hidden messages embedded.

Note that in order to make the competition more realistic the length of hidden messages (the payload) will not be provided. The only available information on the test set is:

1. Each embedding algorithm is used with the same probability.
2. The payload (message length) is adjusted such that the "difficulty" is approximately the same regardless the content of the image. Images with smooth content are used to hide shorter messages while highly textured images will be used to hide more secret bits. The payload is adjusted in the same manner for testing and training sets.
3. The average message length is 0.4 bit per non-zero AC DCT coefficient.
4. The images are all compressed with one of the three following JPEG quality factors: 95, 90 or 75.

- `Cover/` contains 75k unaltered images meant for use in training.
- `JMiPOD/` contains 75k examples of the JMiPOD algorithm applied to the cover images.
- `JUNIWARD/` contains 75k examples of the JUNIWARD algorithm applied to the cover images.
- `UERD/` contains 75k examples of the UERD algorithm applied to the cover images.
- `Test/` contains 5k test set images. These are the images for which you are predicting.

## 사용한 방법

### SEED setting
프로그램 재현성을 위해 랜덤 시드 고정
``` python
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)
```

### transformation function
albumentations을 이용하되 flip & resize만 이용. 그 이유는 steganography 기법을 통해 이미지에 메세지를 숨기게 되면 픽셀 값이 최대한 보존되는 것이 중요하므로 blur, brightness 등과 같은 변환은 사용하지 않음.
``` python
# 2. Make transform funcitons
def get_train_transforms():
    return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)

def get_valid_transforms():
    return A.Compose([
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)
```

### Image read function `JPEGdecompressYCbCr_v3`
주파수 영역에서의 steganography 기법은 다음과 같은 과정에서 이뤄진다.
1. 이미지를 JPEG 압축 하는 과정에서 메세지를 숨긴다.
2. JPEG 압축시 이미지를 YCbCr 색공간에서 각 채널별로 DCT(Discrete Cosine Transform)을 수행하고 양자화를 거친다.
3. 양자화 된 테이블에서 DC 계수에 steganography 기법을 적용시켜 계수 값을 변화
4. encoding 작업을 통해 jpeg 이미지 저장.

따라서 단순히 `cv2.imread`를 이용하여 이미지를 읽으면 int type으로 저장되어 많은 정보가 소실되므로 직접 JPEG decoding 작업을 통해 YCbCr 채널의 float type 데이터로 이미지 로드

### Label Smoothing
메세지가 숨겨진 stego 이미지라 하더라도 모든 픽셀마다 숨겨져있지 않고 일반 cover 이미지의 성격도 가지고 있다. 따라서 label을 만들 때 전체적으로 smoothing하게 두어 학습을 수행시켰다.
``` python
class LabelSmoothing(nn.Module):
    def __init__(self, smoothing = 0.05):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim = -1)

            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)
    
            smooth_loss = -logprobs.mean(dim=-1)

            loss = self.confidence * nll_loss + self.smoothing * smooth_loss

            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)
```
