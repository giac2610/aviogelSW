
import { Component } from '@angular/core';
import { AlertController, ToastController } from '@ionic/angular';

@Component({
  selector: 'tutorial-page',
  templateUrl: './tutorial.page.html',
  styleUrls: ['./tutorial.page.scss'],
  standalone: false
})
export class TutorialPage {

  constructor(private toastController: ToastController, private alertController: AlertController){}

  steps = ["Start", "Step 1", "Step 2", "Step 3", "Finish", "end", "end2"];
  currentStep = 0;
  extruderBool = false;
  toastMessage = "null";

  nextStep() {
    if (this.currentStep < this.steps.length - 1) {
      this.currentStep++;
    }
  }

  prevStep() {
    if (this.currentStep > 0) {
      this.currentStep--;
    }
  }

  extruderToggle(type: 'play' | 'pause'){
    // inviare una POST al backend per poi avviare l'erogazione
    this.extruderBool = !this.extruderBool;
    this.presentToast(type)
    if(type == 'play'){
      // start counter
      setTimeout(() => {
        this.showAlert();
      }, 5000);

    }
  }

  async presentToast(messageType: string) {
    
    if(messageType=='play'){

      this.toastMessage = "Avviata l'estrusione";
    }
    if(messageType == 'pause'){
      this.toastMessage = "estrusione ferma"
    }
    const toast = await this.toastController.create({
      message: this.toastMessage,
      duration: 1500,
    });

    await  toast.present();
  }

  async showAlert(){
    const alert = await this.alertController.create({
      header: 'Sei ancora li?',
      cssClass: 'custom-alert',
      message: 'Se è fuoriuscito liquido in maniera costante dall\'ugello puoi continuare, altrimenti clicca annulla',
      buttons: [
        {
          text: 'annulla',
          role: 'cancel',
          cssClass: 'alert-button-cancel',
          handler: () => {
            console.log('Alert canceled');
          },
        },
        {
          text: 'OK',
          role: 'continua',
          cssClass: 'alert-button-confirm',
          handler: () => {
            console.log('Alert confirmed');
          },
        },
      ]
    });

    await alert.present();
  }
}