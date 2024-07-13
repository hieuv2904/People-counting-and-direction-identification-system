from django.db import models

# Create your models here.
class Alert_log(models.Model):
    time = models.DateTimeField(auto_now_add=True, null=True)
    alert = models.CharField(max_length=30)
    camera_number = models.IntegerField()
    clip_link = models.CharField(max_length=300, null=True)