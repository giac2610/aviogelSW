import { Component, inject } from '@angular/core';
import { AlertController, ModalController, RefresherCustomEvent } from '@ionic/angular';
import { Router} from '@angular/router';
import { RestAPIfromDjangoService, User } from '../../services/rest-apifrom-django.service';
import { Keyboard } from '@capacitor/keyboard';
import { LedService } from '../../services/led.service';
import { EditUserModalComponent } from '../../components/edit-user-modal/edit-user-modal.component';

@Component({
  selector: 'app-home',
  templateUrl: 'home.page.html',
  styleUrls: ['home.page.scss'],
  standalone: false,
})
export class HomePage {
  isLoading = true;
  users: User[] = [];
  isModalOpen = false;
  requiredTap: number = 5;
  numberOfTap = 0;
  newUserData ={
    name: '',
    gender: '',
    isExpert:  false
  };

  constructor(
    private usersService: RestAPIfromDjangoService,  
    private modalCtrl: ModalController, 
    private router: Router,
    private ledService: LedService,
    private alertCtrl: AlertController
  ) {}
  
  ngOnInit() {
    this.ledService.startWaveEffect().subscribe({
      next: (response) => console.log('Wave effect response:', response), // Stampa la risposta JSON
      error: (err) => console.error('Error starting wave effect:', err)
    });
    this.loadUsers();
  }
  loadUsers() {
    const interval = setInterval(() => {
      this.usersService.getUsers().subscribe((data: User[]) => {
        if (data.length > 0) {
          this.users = data;
          this.isLoading = false; // I dati sono stati caricati
          clearInterval(interval); // Ferma il controllo
        } else {
          console.log("Nessun dato disponibile, riprovo tra 5 secondi...");
        }
      }, (error) => {
        console.error("Errore durante il caricamento degli utenti:", error);
      });
    }, 5000); // Controlla ogni 5 secondi
  }

addUser(name: string, gender: string = 'male', expertUser: boolean = false) {
  let isExpertStr: string = expertUser ? "True" : "False";
  this.usersService.addUser(name, gender, isExpertStr).subscribe(newUser => {
    this.users.push(newUser);
  });
}

navigateNextPage(user: User){
  // fare set di user.id nel backend per raccogliere dati
  this.usersService.setCurrentUser(user)
  // navigazione in base all'attributo
  user.expertUser ? this.router.navigate(['/expert']) : this.router.navigate(['/tutorial'])

}

enterSetup(){
  this.numberOfTap++;
  // console.log(this.numberOfTap)
  if(this.numberOfTap == this.requiredTap){
    this.router.navigate(['/setup'])
    this.numberOfTap = 0;
  }
}

async openAddUserModal() {
  const modal = await this.modalCtrl.create({
    component: EditUserModalComponent,
    componentProps: {
      user: { name: '', gender: 'male', expertUser: false, avatar_url: '', id: null }
    }
  });
  modal.onDidDismiss().then(result => {
    if (result.data && result.data.action === 'save') {
      // Chiamata al service per aggiungere l’utente
      this.addUser(
        result.data.user.name,
        result.data.user.gender,
        result.data.user.expertUser
      );
    }
  });
  await modal.present();
}
async openEditUserModal(user: User, event: Event) {
  event.stopPropagation(); // Evita il click sulla card
  const modal = await this.modalCtrl.create({
    component: EditUserModalComponent,
    componentProps: { user }
  });
  modal.onDidDismiss().then(result => {
    if (result.data) {
      if (result.data.action === 'save') {
        // Aggiorna l'utente (implementa updateUser nel service se necessario)
        // this.usersService.updateUser(result.data.user).subscribe(...);
        this.usersService.modifyUser(result.data.user).subscribe({
          next: (response) => {
            console.log('User updated successfully', response);
          },
          error: (error) => {
            console.error('Error updating user', error);
          }
        })
        Object.assign(user, result.data.user); // Aggiorna localmente
      } else if (result.data.action === 'delete') {
        this.deleteUser(user);
      }
    }
  });
  await modal.present();
}

  deleteUser(user: User) {
    // Qui dovrai implementare la chiamata API per eliminare l'utente
    // Esempio:
    this.usersService.deleteUser(user.id).subscribe({
      next: () => {
        this.users = this.users.filter(u => u.id !== user.id);
      },
      error: (err) => {
        console.error('Errore durante l\'eliminazione:', err);
      }
    });
  }
}