1) This project is implemented using Python 2.7.10 and Django web framework and Mysql database.
2) Install Django 1.9.5
3)Setup XAMPP
	run the following commands:
	a) "python manage.py makemigrations"
	b) "python manage.py migrate"
	this will create the mysql tables in the database "stocks".
5) to Run the sytem server:
	a) Navigate to directory /stock-prediction/stockprediction/
	b) Run the command "python manage.py runserver"
----------realtime
		This folder is the main base directory in the system. It contains the most important .py files like:
			urls.py it has main URL mapping of the system. Maps all the urls to their URIs.

----------Manage.py: It is main file in the system.


Directory structure:

Stock-Prediction-Systems - Main folder



----------realtime
		This folder is the main base directory in the system. It contains the most important .py files like:
			settings.py Contains main settings for system. Go and put your mysql password here.
			urls.py it has main URL mapping of the system. Maps all the urls to their URIs.
----------Manage.py: It is main file in the system.
	

----------Templates- Contains all of the HTML templates of the system which is inside realtime.
		index.html
		prediction.html
		news.html
		search.html
		contact.html
		and all other html templates for graph plots.

----------Static- Contains all the static files for the system like images, Javascript files, CSS files, fonts.
 
----------- main App in our system.
	
		urls.py- maps urls of this app
		views.py Contains all the functions to create views on the webpage.


