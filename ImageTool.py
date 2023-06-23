"""
Image tool to generate images based of the given input
"""

# import libraries and required classes
from langchain.tools import BaseTool
import openai
import os

desc = (
    "use this tool when you create an image of any sort."
    "To use the tool, provide a string that describes what the image should contain/look like."
    "What is returned will be either 'success', or 'error'."
    "In the case of 'success', your final answer should be 'Image successfully generated'"
    "Include the following in the description for the image:"
    "A description of the subject in detail,"
    "A description of the background (Ex. 'in times Square', 'buildings in the background'),"
    "The art style of the image (Ex. 'cubist', 'Tim Burton'),"
    "The mood of the image (Ex. 'bright', 'gorgeous')."
)
  
class ImageTool(BaseTool):
    name = "Image generator"
    description = desc

    # OpenAI API key
    openai.api_key = os.environ["OPENAI_API_KEY"]

    def _run(self, query:str):
        try:
            # Generate an image
            response = openai.Image.create(
                prompt=query,
                n=1,
                size="1024x1024",
                response_format="url"
            )

            # print the URL of the generated image
            print(response["data"][0]["url"])

            # print success to show that the image was successfully generated
            return 'success'
        except:
            return 'error'
        
    
    def _arun(self, query:str):
        raise NotImplementedError("This tool does not support async")
  
