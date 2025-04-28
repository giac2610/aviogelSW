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
  globalGranularity: number = 1; // Variabile globale per la granularità degli input numerici
  isGreyscale: boolean = false; // Variabile per gestire la visualizzazione in greyscale
  isThreshold: boolean = false; // Variabile per gestire la visualizzazione threshold
  thresholdStreamUrl: string = this.configService.getThresholdStreamUrl();
  greyscaleStreamUrl: string = this.configService.getGreyscaleStreamUrl();
  normalStreamUrl: string = 'http://localhost:8000/camera/stream/';

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

    // this.presentToast('Caricamento in corso...', 'primary');
    // console.log('test');
  }

  loadConfig() {
    this.configService.getSettings().subscribe((data: Settings) => {
      this.settings = data;
      this.isLoading = false;
    });
  }

  goBack(){
    this.router.navigate(['/home']).then(() => {
      window.location.reload();
    });
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
      const body = {
        "targets": {
            [motor]: distance,
        }
      };
      console.log("request: ",body);
      this.motorsService.moveMotor(body).subscribe({
        next: (response) => {
          // Mostra il messaggio JSON restituito dal backend
          this.presentToast(`Successo: ${response.status} - Target: ${JSON.stringify(response.targets)}`, 'success');
        },
        error: (error) => {
          // Mostra l'errore JSON restituito dal backend
          const errorMessage = error.error.detail || error.error.error || error.message;
          this.presentToast(`Errore: ${errorMessage}`, 'danger');
        }
      });
    } else {
      this.presentToast(`Posizione non ammessa per il motore ${motor}`, 'danger');
    }
  }
  
  goHome(motor: "syringe" | "extruder" | "conveyor"){
    this.goToPosition(motor, -this.positions[motor]);
    this.positions[motor] = 0;
  }

  stopMotors() {
    this.motorsService.stopMotor().subscribe({
      next: (response) => {
        // Mostra il messaggio dal backend
        this.presentToast(`Successo: ${response.status}`, 'success');
      },
      error: (error) => {
        // Mostra l'errore dal backend
        this.presentToast(`Errore: ${error.error.detail || error.message}`, 'danger');
      }
    });
  }

  saveSettings() {
    // console.log("Dati inviati al backend:", this.settings); // Log dei dati
    console.log("Dati inviati al backend:", this.settings); // Log dei dati
    this.motorsService.saveSettings(this.settings).subscribe({
    // this.configService.updateSettings(this.settings).subscribe({
      next: (response) => {
        console.log('Impostazioni salvate:', response);
        this.presentToast('Impostazioni salvate con successo', 'success');
      },
      error: (error) => {
        // console.error('Errore durante il salvataggio delle impostazioni:', error);
        this.presentToast('Errore durante il salvataggio delle impostazioni', 'danger');
      }
    });
  }

  updateCameraSettings() {
    const updatedSettings: Settings = {
      ...this.settings,
      camera: this.settings.camera
    };
    this.configService.updateSettings(updatedSettings).subscribe({
      next: (response) => {
        console.log('Impostazioni aggiornate:', response);
        this.presentToast('Impostazioni aggiornate con successo', 'success');
      },
      error: (error) => {
        console.error('Errore durante l\'aggiornamento delle impostazioni:', error);
        this.presentToast('Errore durante l\'aggiornamento delle impostazioni', 'danger');
      }
    });
  }

  toggleGreyscaleView() {
    console.log('Greyscale view:', this.isGreyscale ? 'Enabled' : 'Disabled');
  }

  toggleThresholdView() {
    console.log('Threshold view:', this.isThreshold ? 'Enabled' : 'Disabled');
  }

  async presentToast(message: string, color: string = 'success') {
    const toast = await this.toastController.create({
      message: message,
      duration: 1400,
      icon: color === 'success' ? 'checkmark-circle' : 'alert-circle',
      color: color
    });
    await toast.present();
  }


  // async presentPositionToast(message: string) {
  //   const toast = await this.toastController.create({
  //     message: message,
  //     duration: 1400,
  //     icon: 'alert-circle',
  //     color: 'danger'
  //   });
  //   await toast.present();
  // }
}
