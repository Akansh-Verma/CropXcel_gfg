from django.shortcuts import render, HttpResponse
from .predict import Predict
import tempfile
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import os
from .forms import ImageForm
from .models import Image

# Create your views here.


def detection(request):
    if request.method == 'POST' and request.FILES['image']:
        # Get uploaded image from request
        image = request.FILES['image']

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(image.read())
            temp_file_path = temp_file.name

        # Get prediction
        predictor = Predict(
            'D:/GithubDesktop/CropXcel_gfg/cropxcel/prediction/CropXcel.h5')

        # For deploy
        # predictor = Predict('cropxcel/prediction/CropXcel.h5')
        predicted_class = predictor.predict(temp_file_path)

        # Save image
        new_image = Image(image=image)
        new_image.save()

        # Render template with results
        context = {
            'predicted_class': predicted_class,
            'uploaded_file_url':  new_image.image.url,
        }
        return render(request, 'detection.html', context)

    # Render empty form if request is GET
    else:
        form = ImageForm()
        return render(request, 'detection.html', {'form': form})
