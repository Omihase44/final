from django.db import models


class Admin_Details(models.Model):
    Username = models.CharField(default=None,max_length=100)
    Password = models.CharField(default=None,max_length=100)
    
    class Meta:
        db_table = 'Admin_Details'
                    




class User_Details(models.Model):
    Name  = models.CharField(default=None,max_length=500)
    Age = models.CharField(default=None,max_length=500)
    Contact = models.CharField(default=None,max_length=500)
    Email = models.CharField(default=None,max_length=500)
    Username = models.CharField(default=None,max_length=500)
    Image1 = models.ImageField(upload_to='img/images',default=None)
    Class_detected = models.CharField(default=None,max_length=500)
    Symptoms= models.CharField(default=None,max_length=500)
    Treatment= models.CharField(default=None,max_length=500)


    class Meta:
        db_table = 'User_Details'







