import glob
import cv2


def loadSingleImageWithPath(path):
    image = cv2.imread(path)
    width = int(image.shape[1] * 0.2)
    height = int(image.shape[0] * 0.2)
    dim = (width, height)
    ImagesMiniatures.append(cv2.resize(image, dim))
    ImagesPaths.append(path)

def loadImagesPathsInFolder(folderPath):
    rawImagesPaths = glob.glob(folderPath)
    for path in rawImagesPaths:
        ImagesPaths.append(path.replace('\\', '/'))
    print(ImagesPaths)

def loadImagesInFolder():
    first_id = len(ImagesMiniatures)
    last_id = len(ImagesPaths)
    for current_id in range(first_id, last_id):
        path = ImagesPaths[current_id]
        image = cv2.imread(path)
        width = int(image.shape[1] * 0.2)
        height = int(image.shape[0] * 0.2)
        dim = (width, height)
        ImagesMiniatures.append(cv2.resize(image, dim))

def deleteImage(id):
    del ImagesPaths[id]
    del ImagesMiniatures[id]

def deleteAllImages():
    ImagesPaths.clear()
    ImagesMiniatures.clear()

def displayImagesAndPaths():
    id = 0
    for path in ImagesPaths:
        image = cv2.imread(path)
        cv2.imshow("image", ImagesMiniatures[id])
        print("Press esc to exit or any other button to continue...")
        key = cv2.waitKey()
        id += 1
        if key == 27:
            break
    cv2.destroyAllWindows()
    print(ImagesPaths)

if __name__ == "__main__":
    ImagesPaths = []
    ImagesMiniatures = []
    loadSingleImageWithPath('images/test/3096.jpg')
    loadSingleImageWithPath('images/test/8023.jpg')
    loadSingleImageWithPath('images/test/12084.jpg')
    loadSingleImageWithPath('images/test/14037.jpg')
    loadSingleImageWithPath('images/test/16077.jpg')
    # loadImagesPathsInFolder('images/test/*.jpg')
    # loadImagesInFolder(ImagesPaths)
    displayImagesAndPaths()
    deleteImage(3)
    displayImagesAndPaths()
    deleteAllImages()
    displayImagesAndPaths()