
import { Component } from '@angular/core';
import { AlertController, ToastController } from '@ionic/angular';
import { MotorsControlService } from 'src/app/services/motors-control.service';

@Component({
  selector: 'tutorial-page',
  templateUrl: './tutorial.page.html',
  styleUrls: ['./tutorial.page.scss'],
  standalone: false
})
export class TutorialPage {

  constructor(private toastController: ToastController, private alertController: AlertController, private motorsService: MotorsControlService,){}

  steps = ["Start", "Step 1", "Step 2", "Step 3", "Finish", "end", "end2"];
  currentStep = 0;
  extruderBool = false;
  toastMessage = "null";

  nextStep() {
    if(this.currentStep == 4){
      this.extruderBool = false;
    }
    if (this.currentStep < this.steps.length - 1) {
      this.currentStep++;
    }
  }

  prevStep() {
    if(this.currentStep == 4){
      this.extruderBool = false;
    }
    if (this.currentStep > 0) {
      this.currentStep--;
    }
  }

  extruderToggle(type: 'play' | 'pause'){
    // SAREBBE SYRINGE IN REALTA
    // inviare una POST al backend per poi avviare l'erogazione
    this.extruderBool = !this.extruderBool;
    this.presentToast(type)
    if(type == 'play'){
      // start counter
      setTimeout(() => {
        this.showAlert();
      }, 5000);
      if(this.extruderBool){      
        this.motorsService.syringeExtrusionStart().subscribe({
          next: (response) => console.log('Syringe extrusion started:', response),
          error: (err) => console.error('Error starting syringe extrusion:', err)
        });
      } else {
        this.motorsService.stopMotor().subscribe({
          next: (response) => console.log('Syringe extrusion stopped:', response),
          error: (err) => console.error('Error stopping syringe extrusion:', err)
        });
      }
    }
  }

  async presentToast(messageType: string) {
    
    if(messageType=='play'){

      this.toastMessage = "Extrution Started";
    }
    if(messageType == 'pause'){
      this.toastMessage = "Estruxion Paused";
    }
    const toast = await this.toastController.create({
      message: this.toastMessage,
      duration: 1500,
    });

    await  toast.present();
  }

  async showAlert(){
    const alert = await this.alertController.create({
      header: 'Are you still there?',
      cssClass: 'custom-alert',
      message: 'If liquid is flowing steadily from the nozzle, you can continue. Otherwise, click cancel.',
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
            this.extruderBool = false;
          },
        },
      ]
    });

    await alert.present();
  }
}