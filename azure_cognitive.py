# <snippet_imports>
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import TextOperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import TextRecognitionMode
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

from array import array
import os
from PIL import Image
import sys
import time
# </snippet_imports>

'''
Computer Vision Quickstart for Microsoft Azure Cognitive Services. 
Uses local and remote images in each example.
Prerequisites:
    - Install the Computer Vision SDK:
      pip install --upgrade azure-cognitiveservices-vision-computervision
References:
    - SDK: https://docs.microsoft.com/en-us/python/api/azure-cognitiveservices-vision-computervision/azure.cognitiveservices.vision.computervision?view=azure-python
    - Documentaion: https://docs.microsoft.com/en-us/azure/cognitive-services/computer-vision/index
    - API: https://westus.dev.cognitive.microsoft.com/docs/services/5adf991815e1060e6355ad44/operations/56f91f2e778daf14a499e1fa
'''

'''
Quickstart variables
These variables are shared by several examples
'''
# Images used for the examples: Describe an image, Categorize an image, Tag an image, 
# Detect faces, Detect adult or racy content, Detect the color scheme, 
# Detect domain-specific content, Detect image types, Detect objects
local_image_path = "resources/example1.jpg"
# <snippet_remoteimage>
remote_image_url = "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files/master/ComputerVision/Images/landmark.jpg"
# </snippet_remoteimage>
'''
END - Quickstart variables
'''

'''
Authenticate
Authenticates your credentials and creates a client.
'''
# <snippet_vars>
# Add your Computer Vision subscription key to your environment variables.
if 'COMPUTER_VISION_SUBSCRIPTION_KEY' in os.environ:
    subscription_key = os.environ['COMPUTER_VISION_SUBSCRIPTION_KEY']
else:
    print("\nSet the COMPUTER_VISION_SUBSCRIPTION_KEY environment variable.\n**Restart your shell or IDE for changes to take effect.**")
    sys.exit()
# Add your Computer Vision endpoint to your environment variables.
if 'COMPUTER_VISION_ENDPOINT' in os.environ:
    endpoint = os.environ['COMPUTER_VISION_ENDPOINT']
else:
    print("\nSet the COMPUTER_VISION_ENDPOINT environment variable.\n**Restart your shell or IDE for changes to take effect.**")
    sys.exit()
# </snippet_vars>

# <snippet_client>
computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))
# </snippet_client>
'''
END - Authenticate
'''

'''
Batch Read File, recognize handwritten text - local
This example extracts text from a handwritten local image, then prints results.
This API call can also recognize printed text (shown in next example, Batch Read File - remote).
'''
print("===== Batch Read File - local =====")
# Get image of handwriting
local_image_handwritten_path = local_image_path
# Open the image
local_image_handwritten = open(local_image_handwritten_path, "rb")

# Call API with image and raw response (allows you to get the operation location)
recognize_handwriting_results = computervision_client.batch_read_file_in_stream(local_image_handwritten, raw=True)
# Get the operation location (URL with ID as last appendage)
operation_location_local = recognize_handwriting_results.headers["Operation-Location"]
# Take the ID off and use to get results
operation_id_local = operation_location_local.split("/")[-1]

# Call the "GET" API and wait for the retrieval of the results
while True:
    recognize_handwriting_result = computervision_client.get_read_operation_result(operation_id_local)
    if recognize_handwriting_result.status not in ['NotStarted', 'Running']:
        break
    time.sleep(1)

# Print results, line by line
if recognize_handwriting_result.status == TextOperationStatusCodes.succeeded:
    for text_result in recognize_handwriting_result.recognition_results:
        for line in text_result.lines:
            print(line.text)
            print(line.bounding_box)
print()
'''
END - Batch Read File - local
'''

# '''
# Recognize Printed Text with OCR - local
# This example will extract, using OCR, printed text in an image, then print results line by line.
# '''
# print("===== Detect Printed Text with OCR - local =====")
# # Get an image with printed text
# local_image_printed_text_path = local_image_path
# local_image_printed_text = open(local_image_printed_text_path, "rb")

# ocr_result_local = computervision_client.recognize_printed_text_in_stream(local_image_printed_text)
# for region in ocr_result_local.regions:
#     for line in region.lines:
#         print("Bounding box: {}".format(line.bounding_box))
#         s = ""
#         for word in line.words:
#             s += word.text + " "
#         print(s)
# print()
# '''
# END - Recognize Printed Text with OCR - local
# '''

print("End of Computer Vision quickstart.")