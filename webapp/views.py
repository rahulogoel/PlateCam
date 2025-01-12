from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .deeplearn import OCR  

def upload_image(request):
    if request.method == 'POST' and request.FILES.get('vehicle_image'):
        uploaded_file = request.FILES['vehicle_image']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        uploaded_file_url = fs.url(filename)

        extracted_text, plate_image_url, bounded_plate_image_url = OCR(fs.path(filename))

        return render(request, 'index.html', {
            'uploaded_image_url': uploaded_file_url,
            'plate_image_url': plate_image_url,
            'bounded_plate_image_url': bounded_plate_image_url,
            'extracted_text': extracted_text,
            'processed': True,
        })

    return render(request, 'index.html', {'processed': False})
