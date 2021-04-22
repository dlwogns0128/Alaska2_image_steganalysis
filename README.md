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
- `iPOD/` contains 75k examples of the JMiPOD algorithm applied to the cover images.
- `JUNIWARD/` contains 75k examples of the JUNIWARD algorithm applied to the cover images.
- `UERD/` contains 75k examples of the UERD algorithm applied to the cover images.
- `Test/` contains 5k test set images. These are the images for which you are predicting.

## 사용한 방법
