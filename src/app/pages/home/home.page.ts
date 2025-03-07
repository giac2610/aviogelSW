import { Component, inject } from '@angular/core';
import { AlertController, ModalController, RefresherCustomEvent } from '@ionic/angular';
import { Router} from '@angular/router';
import { RestAPIfromDjangoService, User } from '../../services/rest-apifrom-django.service';
import { Keyboard } from '@capacitor/keyboard';

@Component({
  selector: 'app-home',
  templateUrl: 'home.page.html',
  styleUrls: ['home.page.scss'],
  standalone: false,
})
export class HomePage {


  users: User[] = [];
  isModalOpen = false;
  isExpert = false;

  constructor(private usersService: RestAPIfromDjangoService,  private modalCtrl: ModalController, private router: Router) {}

  
  ngOnInit() {
    this.loadUsers();
  }

  loadUsers() {
    this.usersService.getUsers().subscribe((data: User[]) => {
      this.users = data;
    });


}


// async openAddUserModal() {
//   const alert = await this.modalCtrl.create({
//     // header: 'Aggiungi Utente',
//     inputs: [
//       { name: 'name', type: 'text', placeholder: 'Nome' },
//       { type: 'radio', label: 'Maschio', value: 'male' },
//       { type: 'radio', label: 'Femmina', value: 'female' }
//     ],
//     buttons: [
//       { text: 'Annulla', role: 'cancel' },
//       { text: 'Aggiungi', handler: (data) => this.addUser(data.name, data.gender, data.expertUser) }
//     ]
//   });


//   await alert.present();
// }

addUser(name: string, gender: string, expertUser: string) {
  this.usersService.addUser(name, gender, expertUser).subscribe(newUser => {
    this.users.push(newUser);
  });
}

navigateNextPage(user: User){
  // fare set di user.id nel backend per raccogliere dati

  // navigazione in base all'attributo
  user.expertUser ? this.router.navigate(['/expert']) : this.router.navigate(['/tutorial'])

}

}