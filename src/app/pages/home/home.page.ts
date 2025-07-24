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
  ) {}
  
  ngOnInit() {
    this.ledService.startWaveEffect().subscribe({
      next: (response) => console.log('Wave effect response:', response),
      error: (err) => console.error('Error starting wave effect:', err)
    });
    this.loadUsers();
  }
  loadUsers() {
    const interval = setInterval(() => {
      this.usersService.getUsers().subscribe((data: User[]) => {
        if (data.length > 0) {
          this.users = data;
          this.isLoading = false;
          clearInterval(interval);
        } else {
          console.log("Nessun dato disponibile, riprovo tra 5 secondi...");
        }
      }, (error) => {
        console.error("Errore durante il caricamento degli utenti:", error);
      });
    }, 5000);
  }

addUser(name: string, gender: string = 'male', expertUser: boolean = false) {
  let isExpertStr: string = expertUser ? "True" : "False";
  this.usersService.addUser(name, gender, isExpertStr).subscribe(newUser => {
    this.users.push(newUser);
  });
}

navigateNextPage(user: User){
  this.usersService.setCurrentUser(user)
  user.expertUser ? this.router.navigate(['/expert']) : this.router.navigate(['/tutorial'])

}

enterSetup(){
  this.numberOfTap++;
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
    if (result.data && result.data.action === 'save'){
      this.addUser(
        result.data.user.name,
        result.data.user.gender,
        result.data.user.expertUser
      );
    }
  });
  await modal.present();
}
async openEditUserModal(user: User, event: Event){
  event.stopPropagation();
  const modal = await this.modalCtrl.create({
    component: EditUserModalComponent,
    componentProps: { user }
  });
  modal.onDidDismiss().then(result => {
    if (result.data) {
      if (result.data.action === 'save') {
        this.usersService.modifyUser(result.data.user).subscribe({
          next: (response) => {
            console.log('User updated successfully', response);
          },
          error: (error) => {
            console.error('Error updating user', error);
          }
        })
        Object.assign(user, result.data.user);
      } else if (result.data.action === 'delete') {
        this.deleteUser(user);
      }
    }
  });
  await modal.present();
}

  deleteUser(user: User){
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