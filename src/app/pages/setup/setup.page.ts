import { MotorsControlService } from './../../services/motors-control.service';
import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { ToastController } from '@ionic/angular';
import { SetupAPIService, Settings } from 'src/app/services/setup-api.service';
import { debounceTime, Subject } from 'rxjs';
import { LedService } from 'src/app/services/led.service';
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
  isThreshold: boolean = false; // Variabile per gestire la visualizzazione threshold
  thresholdStreamUrl: string = this.configService.getThresholdStreamUrl();
  normalStreamUrl: string = 'http://localhost:8000/camera/stream/';
  selectedStream: string = 'normal'; // Default stream type
  currentStreamUrl: string = this.normalStreamUrl; // Default stream URL

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

  private cameraSettingsSubject = new Subject<Settings['camera']>();

  constructor(private configService: SetupAPIService, private toastController: ToastController, private motorsService: MotorsControlService, private router: Router, private ledService: LedService) { }

  ngOnInit() {
    this.isLoading = true;
    this.loadConfig();

    // Ascolta i cambiamenti nei settaggi della camera e invia al backend
    this.cameraSettingsSubject.pipe(debounceTime(300)).subscribe((cameraSettings) => {
      this.configService.updateCameraSettings(cameraSettings).subscribe({
        next: (response) => {
          console.log('Impostazioni aggiornate in live:', response);
        },
        error: (error) => {
          console.error('Errore durante l\'aggiornamento delle impostazioni in live:', error);
        }
      });
    });

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
      const targets = { [motor]: distance }; // Correctly structure the targets object
      console.log("request: ", targets);
      this.configService.moveMotor(targets).subscribe({
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
    this.configService.stopMotors().subscribe({
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
    console.log("Dati inviati al backend:", this.settings); // Log dei dati
    this.configService.updateSettings(this.settings).subscribe({
      next: (response) => {
        console.log('Impostazioni salvate:', response);
        this.presentToast('Impostazioni salvate con successo', 'success');
      },
      error: (error) => {
        this.presentToast('Errore durante il salvataggio delle impostazioni', 'danger');
      }
    });
  }

  onCameraSettingChange() {
    this.cameraSettingsSubject.next(this.settings.camera);
  }

  updateCameraSettings() {
    // Metodo manuale per aggiornare i settaggi (opzionale)
    this.configService.updateCameraSettings(this.settings.camera).subscribe({
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

  async presentToast(message: string, color: string = 'success') {
    const toast = await this.toastController.create({
      message: message,
      duration: 1400,
      icon: color === 'success' ? 'checkmark-circle' : 'alert-circle',
      color: color
    });
    await toast.present();
  }

  updateStreamUrl() {
    switch (this.selectedStream) {
      case 'threshold':
        this.currentStreamUrl = `${this.normalStreamUrl}?mode=threshold&keyframe=true`; // Modalità threshold con keyframe
        break;
      case 'normal':
      default:
        this.currentStreamUrl = `${this.normalStreamUrl}?mode=normal&keyframe=true`; // Modalità normale con keyframe
        break;
    }
    console.log(`Stream URL aggiornato a: ${this.currentStreamUrl}`); // Debug log
  }

  onImageError() {
    console.error("Errore durante il caricamento dello streaming."); // Log per debug
    this.presentToast("Errore durante il caricamento dello streaming.", "danger");
  }

  startWaveEffect() {
    this.ledService.startWaveEffect().subscribe({
      next: () => this.presentToast('Wave effect avviato', 'success'),
      error: () => this.presentToast('Errore nell\'avviare il wave effect', 'danger')
    });
  }
  
  startGreenLoading() {
    this.ledService.startGreenLoading().subscribe({
      next: () => this.presentToast('Green loading avviato', 'success'),
      error: () => this.presentToast('Errore nell\'avviare il green loading', 'danger')
    });
  }
  
  startYellowBlink() {
    this.ledService.startYellowBlink().subscribe({
      next: () => this.presentToast('Yellow blink avviato', 'success'),
      error: () => this.presentToast('Errore nell\'avviare il yellow blink', 'danger')
    });
  }
  
  startRedStatic() {
    this.ledService.startRedStatic().subscribe({
      next: () => this.presentToast('Red static avviato', 'success'),
      error: () => this.presentToast('Errore nell\'avviare il red static', 'danger')
    });
  }

  stopEffect() {
    this.ledService.stopEffect().subscribe({
      next: () => this.presentToast('Effetto fermato', 'success'),
      error: () => this.presentToast('Errore nel fermare l\'effetto', 'danger')
    });
  }
}
