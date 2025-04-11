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

  constructor(private usersService: RestAPIfromDjangoService,  private modalCtrl: ModalController, private router: Router) {}
  
  ngOnInit() {
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
showDeleteButton(user: User) {
  // Mostra il pulsante di eliminazione per l'utente specificato
  // user.showDeleteButton = true;
  console.log("long press on: ", user)
}

}