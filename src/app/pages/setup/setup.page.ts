import { MotorsControlService } from './../../services/motors-control.service';
import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { ToastController } from '@ionic/angular';
import { SetupAPIService, Settings } from 'src/app/services/setup-api.service';

@Component({
  selector: 'app-setup',
  templateUrl: './setup.page.html',
  styleUrls: ['./setup.page.scss'],
  standalone: false,
})
export class SetupPage implements OnInit {
  selectedMotor: string = 'none';
  settings!: Settings;
  testMode: boolean = false;
  isLoading = true; 

 // Oggetto che contiene le posizioni dei motori
positions: { [key in "syringe" | "extruder" | "conveyor"]: number } = {
  syringe: 0,
  extruder: 0,
  conveyor: 0
};

travels: { [key in "syringe" | "extruder" | "conveyor"]: number } = {
  syringe: 0,
  extruder: 0,
  conveyor: 0
};
  SetupAPIService: any;

  constructor(private configService: SetupAPIService, private toastController: ToastController, private motorsService: MotorsControlService, private router: Router) { }

  ngOnInit() {
    this.isLoading = true;
    this.loadConfig();
  }

  loadConfig() {
    this.configService.getSettings().subscribe((data: Settings) => {
      this.settings = data;
      this.isLoading = false;
    });
  }

  goBack(){
    this.router.navigate(['/home']);
  }

  changeMode(mode: string){
    if(mode=="test"){;
      this.testMode = true
    }
    if(mode=="edit"){;
      this.testMode = false
    }
  }

  closeMotors(){
    this.selectedMotor = 'none';
  }
  // Funzione per far muovere il motore
  // selezionando il motore e la distanza
  goToPosition(motor: "syringe" | "extruder" | "conveyor", distance: number) {
    const maxTravel = this.settings.motors[motor].maxTravel;

    if (this.positions[motor] + distance <= maxTravel) {
      this.positions[motor] += distance; 
      console.log(`Nuova posizione di ${motor}:`, this.positions[motor]);
      const body = {
        "targets": {
            [motor]: distance,
        }
      }
      console.log(body);
      this.motorsService.moveMotor(body).subscribe({
        next: (response) => {
          console.log(`Risposta API per ${motor}:`, response);
        },
        error: (error) => {
          console.error(`Errore API per ${motor}:`, error);
        }
      })
    } else {
      // console.warn(`${motor} ha raggiunto il limite massimo.`);
      this.presentPositionToast( `posizione non ammessa per: ${motor} motor`)
    }
  }

  goHome(motor: "syringe" | "extruder" | "conveyor"){
    this.goToPosition(motor, -this.positions[motor]);
    this.positions[motor] = 0;
  }

  stopMotors(){

    this.motorsService.stopMotor().subscribe({
      next: (response) => {
        console.log("Risposta API: ", response);
      },
      error: (error) => {
        console.error("Errore API: ", error);
      }
    })
  }

  saveSettings() {
    // console.log("Dati inviati al backend:", this.settings); // Log dei dati
    this.configService.updateSettings(this.settings).subscribe({
      next: (response) => {
        // console.log('Impostazioni salvate:', response);
        this.presentToast('Impostazioni salvate con successo', 'success');
      },
      error: (error) => {
        // console.error('Errore durante il salvataggio delle impostazioni:', error);
        this.presentToast('Errore durante il salvataggio delle impostazioni', 'danger');
      }
    });
  }
  async presentToast(message: string, color: string = 'success') {
    const toast = await this.toastController.create({
      message: message,
      duration: 1400,
      icon: 'checkmark-circle',
      color: color
    }); 
    await toast.present();
  }


  async presentPositionToast(message: string) {
    const toast = await this.toastController.create({
      message: message,
      duration: 1400,
      icon: 'alert-circle',
      color: 'danger'
    });
    await toast.present();
  }
}
