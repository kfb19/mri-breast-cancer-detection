
# BreastWise API

BreastWise API is a Django application for the detection of Breast Cancer in MRI scans. It uses two VGG19 models with different input channel types to detect cancer on a scan slice-by-slice basis. 
    
## Run Locally

Go to the project directory. 

```bash
  cd breastwise
```

Add your secret key to .env.prod by generating a string of length 50 at this site: http://www.unit-conversion.info/texttools/random-string-generator/

Run this command to run the API in docker. 

```bash
  docker-compose -f docker-compose.yml up -d --build

```

Requests can then be made to the upload endpoint at 127.0.0.1:8000/upload/ .

## Usage/Examples

The API request body should contain a folder called "series.zip". This folder should contain these folders: 
- pre
- 1st_pass
- 2nd_pass
- 3rd_pass
All three folders should contain the same number of DICOM breast MRI images for the relevant series type. The naming should match, ie each folder should contain a file called 001.dcm. 

The response will be in the form of an array, containing information on the slices that contain cancer. For example, [10, 11, 12] would indicate the 10th, 11th and 12th slices contain a tumour. 
## Tech 

Django, Nginx, and Gunicorn work together to create a powerful and efficient web application stack. Let's break down the roles of each component and how they interact with each other:

Django is a high-level Python web framework that enables the rapid development of secure and maintainable websites. It follows the Model-View-Controller (MVC) architectural pattern, which allows for a clear separation of concerns between different parts of an application. Django manages the backend logic, handles database operations, and provides a templating system to generate HTML.

Gunicorn (Green Unicorn) is a Python Web Server Gateway Interface (WSGI) HTTP server. It's a middleman between your Django application and the web server. Gunicorn is responsible for managing multiple worker processes, each of which runs an instance of your Django application. These worker processes handle incoming HTTP requests, execute the corresponding Django views, and return the response. Gunicorn makes it easier to manage the resources used by your Django application, enabling it to handle multiple requests concurrently and efficiently.

Nginx (pronounced "engine-x") is a high-performance web server and reverse proxy server. As a web server, it serves static files (like images, CSS, and JavaScript) directly to the client, which reduces the load on the Django application. As a reverse proxy, it receives incoming HTTP requests from clients (e.g., web browsers) and forwards them to the appropriate Gunicorn worker processes. Nginx can also handle SSL/TLS termination, load balancing, and caching, among other features.

Here's how they work together:

A client sends an HTTP request to your web application.
Nginx receives the request and checks if it's for a static file. If it is, Nginx serves the file directly. If not, it forwards the request to Gunicorn.
Gunicorn receives the request, selects a worker process, and passes the request to the Django application running within that process.
Django processes the request, executes the relevant view function, and generates an HTTP response.
The response is sent back through Gunicorn and Nginx to the client.
In summary, Django handles the application logic and database operations, Gunicorn manages the worker processes running Django instances, and Nginx serves static files and acts as a reverse proxy for handling client requests. This combination provides a scalable and efficient architecture for deploying web applications.

The base image for this stage is python:3.10-buster, which is an official Python 3.10 image based on the Debian Buster distribution.
It sets the working directory to /usr/src/app.
Environment variables are set to disable writing bytecode files (.pyc) and to ensure unbuffered output for Python.
The requirements.txt file is copied into the image.
Pip, PyTorch, torchvision, torchaudio, psycopg2, Django REST framework, Django, and numpy are installed.
The pip wheel command is used to create wheels for the packages listed in requirements.txt. These wheels are stored in the /usr/src/app/wheels directory.
Stage 2: Final
The final stage is responsible for setting up the final application image, which includes the installation of dependencies and copying the source code of the application.

It uses the same base image as the builder stage, python:3.10-buster.
It creates the necessary directories for the application, such as /home/app and its subdirectories.
The working directory is set to $APP_HOME.
The wheels and requirements.txt file from the builder stage are copied to the final stage.
Pip, PyTorch, torchvision, torchaudio, psycopg2, Django REST framework, Django, and numpy are installed again.
The packages in the wheels directory are installed using the pip install --no-cache /wheels/* command.
The entrypoint.prod.sh script is copied to the image and made executable.
The application source code is copied to the $APP_HOME directory.
The ENTRYPOINT instruction is set to execute the entrypoint.prod.sh script when the container starts.	

## Functions 

### post(self, request, *args, **kwargs) 

Defines what happens on a post request - data is unzipped and files are checked. 

### delete_folders()

Deletes temporary folders after use. 

### scan() 

Runs the code for MRI scan analysis, returning an array of scan slices containing cancer. 

### process_single(single_folder) 

Preprocesses data for analysis of fat saturated series images, contained in single_folder. 

### process_scantype(pass1_folder, pass2_folder, pass3_folder)

Preprocesses data for analysis of scantype (Phase 1, Phase 2 & Phase 3) series images, contained in the corresponding folders. 

### normalize(img)

Normalises image pixel values to range [0, 255]. 

### analyse_single()

Puts the fat saturated images through the VGG19 network and returns an array containing the results (0 for negative, 1 for positive). 

### analyse_scantype()

Puts the scantype (Phase 1, Phase 2 & Phase 3) images through the VGG19 network and returns an array containing the results (0 for negative, 1 for positive). 

### average_results(single_results, scantype_results)

Averages the two arrays; if both are 1 then the slice is cancerous, if both are 0 then the slice is non-cancerous, and if they disagree then the nearby slices are looked at and if at least one of those is positive then the slices is defined as positive, as tumour scan slices will be close togther. 
## Running Tests

To run the unit tests tests, run the following command in the breastiwse\app directory.

```bash
  python manage.py test
```


## Support

For support, email kate.belson@hotmail.com .


## License

[MIT](https://choosealicense.com/licenses/mit/)


## Authors

- [Kate Belson](https://github.com/kfb19)

