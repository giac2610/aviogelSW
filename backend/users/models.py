import random
from django.db import models

class UserProfile(models.Model):
    GENDER_CHOICES = [
        ('male', 'Maschio'),
        ('female', 'Femmina'),
        ('other', 'altro')
    ]

    name = models.CharField(max_length=100)
    gender = models.CharField(max_length=10, choices=GENDER_CHOICES, blank=True)
    avatar = models.CharField(max_length=100, blank=True)
    expertUser = models.BooleanField(default=False, blank=True)
    
    def save(self, *args, **kwargs):
        if not self.avatar:
            default_avatars = {
                'male': ['male_1.jpg'],
                'female': ['female_1.jpg', 'female_2.jpg'],
                'other': ['other_1.jpg']
            }
            self.avatar = random.choice(default_avatars.get(self.gender, ['male_1.jpg']))
        super().save(*args, **kwargs)

    def __str__(self):
        return self.name
