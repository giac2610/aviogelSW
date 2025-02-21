import { Component, inject } from '@angular/core';
import { AlertController, RefresherCustomEvent } from '@ionic/angular';
import { Router} from '@angular/router';
import { RestAPIfromDjangoService, User } from '../services/rest-apifrom-django.service';

@Component({
  selector: 'app-home',
  templateUrl: 'home.page.html',
  styleUrls: ['home.page.scss'],
  standalone: false,
})
export class HomePage {
  users: User[] = [];
  
  constructor(private usersService: RestAPIfromDjangoService,  private alertCtrl: AlertController) {}

  ngOnInit() {
    this.loadUsers();
  }

  loadUsers() {
    this.usersService.getUsers().subscribe((data: User[]) => {
      this.users = data;
    });
}


async openAddUserModal() {
  const alert = await this.alertCtrl.create({
    header: 'Aggiungi Utente',
    inputs: [
      { name: 'name', type: 'text', placeholder: 'Nome' },
      { name: 'gender', type: 'radio', label: 'Maschio', value: 'male', checked: true },
      { name: 'gender', type: 'radio', label: 'Femmina', value: 'female' }
    ],
    buttons: [
      { text: 'Annulla', role: 'cancel' },
      { text: 'Aggiungi', handler: (data) => this.addUser(data.name, data.gender) }
    ]
  });


  await alert.present();
}
addUser(name: string, gender: string) {
  this.usersService.addUser(name, gender).subscribe(newUser => {
    this.users.push(newUser);
  });
}


}