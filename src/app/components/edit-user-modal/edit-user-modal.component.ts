import { Component, Input } from '@angular/core';
import { IonicModule, ModalController } from '@ionic/angular';
import { User, RestAPIfromDjangoService } from '../../services/rest-apifrom-django.service';
import { IonHeader, IonInput, IonList, IonContent, IonTitle, IonToolbar, IonIcon, IonButton, IonItem } from "@ionic/angular/standalone";
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-edit-user-modal',
  templateUrl: './edit-user-modal.component.html',
  styleUrls: ['./edit-user-modal.component.scss'],
  imports: [IonicModule, FormsModule],
  standalone: true
})
export class EditUserModalComponent {
  @Input() user!: User;
  name: string = '';
  gender: string = '';
  expertUser: boolean = false;

  constructor(
    private modalCtrl: ModalController,
    private userService: RestAPIfromDjangoService,
  ) {}

  ngOnInit() {
    this.name = this.user.name;
    this.gender = this.user.gender;
    this.expertUser = this.user.expertUser;
  }

  save_user() {
          // Qui puoi chiamare una funzione di update sul service se la implementi
    this.modalCtrl.dismiss({
      action: 'save',
      user: { ...this.user, name: this.name, gender: this.gender, expertUser: this.expertUser }
    });
  }

  delete() {
    this.modalCtrl.dismiss({
      action: 'delete',
      user: this.user
    });
  }

  close() {
    this.modalCtrl.dismiss({ action: 'cancel' });
  }
}