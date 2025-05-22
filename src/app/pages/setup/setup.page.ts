import { MotorsControlService } from './../../services/motors-control.service';
import { Component, OnInit, OnDestroy } from '@angular/core';
import { Router } from '@angular/router';
import { ToastController } from '@ionic/angular';
import { SetupAPIService, Settings } from 'src/app/services/setup-api.service';
import { debounceTime, Subject, interval, Subscription } from 'rxjs';
import { LedService } from 'src/app/services/led.service';

@Component({
  selector: 'app-setup',
  templateUrl: './setup.page.html',
  styleUrls: ['./setup.page.scss'],
  standalone: false,
})
export class SetupPage implements OnInit, OnDestroy {
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

currentSpeeds: { syringe: number; extruder: number; conveyor: number } = {
  syringe: 0,
  extruder: 0,
  conveyor: 0
};

speedPollingSubscription!: Subscription;

  SetupAPIService: any;

  private cameraSettingsSubject = new Subject<Settings['camera']>();

  cameraOrigin = { x: 0, y: 0 };

  constructor(private configService: SetupAPIService, private toastController: ToastController, private motorsService: MotorsControlService, private router: Router, private ledService: LedService) { }

  ngOnInit() {
    this.isLoading = true;
    this.loadConfig();

    // Avvia il polling della velocità in modalità test
    this.startSpeedPolling();

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

  startSpeedPolling() {
    // this.speedPollingSubscription = interval(10).subscribe(() => {
    //   this.configService.getCurrentSpeeds().subscribe({
    //     next: (speeds) => {
    //       this.currentSpeeds = speeds; // Rimosso il cast esplicito
    //     },
    //     error: (error) => {
    //       console.error('Errore durante il polling della velocità:', error);
    //     }
    //   });
    // });
  }

  stopSpeedPolling() {
    if (this.speedPollingSubscription) {
      this.speedPollingSubscription.unsubscribe();
    }
  }

  ngOnDestroy() {
    this.stopSpeedPolling();
  }

  loadConfig() {
    this.configService.getSettings().subscribe((data: Settings) => {
      this.settings = data;
      // Carica x e y della camera se presenti
      if (this.settings.camera && typeof this.settings.camera.origin_x === 'number' && typeof this.settings.camera.origin_y === 'number') {
        this.cameraOrigin.x = this.settings.camera.origin_x;
        this.cameraOrigin.y = this.settings.camera.origin_y;
      }
      this.isLoading = false;

      // Calcola gli stepsPerMm per ogni motore
      ["syringe", "extruder", "conveyor"].forEach((motor) => {
        const motorSettings = this.settings.motors[motor as "syringe" | "extruder" | "conveyor"];
        motorSettings.stepsPerMm = (motorSettings.stepOneRev * motorSettings.microstep) / motorSettings.pitch;
      });
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

    if (maxTravel < 0 || this.positions[motor] + distance <= maxTravel) {
        this.positions[motor] += distance; 
        const targets = { [motor]: distance }; // Correctly structure the targets object
        console.log("request: ", targets);
        this.configService.moveMotor(targets).subscribe({
            next: (response) => {
                // Mostra il messaggio JSON restituito dal backend
                this.presentToast(`Successo: ${response.status} - Target: ${JSON.stringify(response.targets)}`, 'success');
                console.log('Risposta dal backend:', response);
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

  updateMotorHertz(motor: "syringe" | "extruder" | "conveyor") {
    const motorSettings = this.settings.motors[motor];
    motorSettings.hertz = (motorSettings.maxSpeed * motorSettings.stepOneRev * motorSettings.microstep) / motorSettings.pitch;
  }

  saveSettings() {
    // Aggiorna gli hertz e gli stepsPerMm per tutti i motori prima di salvare
    ["syringe", "extruder", "conveyor"].forEach((motor) => {
      const motorSettings = this.settings.motors[motor as "syringe" | "extruder" | "conveyor"];
      motorSettings.hertz = (motorSettings.maxSpeed * motorSettings.stepOneRev * motorSettings.microstep) / motorSettings.pitch;
      motorSettings.stepsPerMm = (motorSettings.stepOneRev * motorSettings.microstep) / motorSettings.pitch;
    });

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

  updateCameraOrigin() {
    // Aggiorna x e y sia localmente che sul backend
    this.settings.camera.origin_x = this.cameraOrigin.x;
    this.settings.camera.origin_y = this.cameraOrigin.y;
    this.configService.setCameraOrigin(this.cameraOrigin.x, this.cameraOrigin.y).subscribe({
      next: (res) => {
        this.presentToast('Origine camera aggiornata', 'success');
      },
      error: () => {
        this.presentToast('Errore aggiornamento origine camera', 'danger');
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

  goToPositionAll(extruderDistance: number, conveyorDistance: number, syringeDistance: number) {
    const targets: { [key: string]: number } = {
      extruder: extruderDistance,
      conveyor: conveyorDistance,
      syringe: syringeDistance
    };
  
    this.configService.moveMotor(targets).subscribe({
      next: (response) => {
        this.presentToast(`Tutti i motori sono stati mossi con successo: ${JSON.stringify(response.targets)}`, 'success');
      },
      error: (error) => {
        const errorMessage = error.error.detail || error.error.error || error.message;
        this.presentToast(`Errore durante il movimento dei motori: ${errorMessage}`, 'danger');
      }
    });
  }

  getTravelValue(motor: string): number {
    return this.travels[motor as "syringe" | "extruder" | "conveyor"] || 0;
  }

  setTravelValue(motor: string, value: number): void {
    this.travels[motor as "syringe" | "extruder" | "conveyor"] = value;
  }

  startSimulation() {
    this.motorsService.simulate().subscribe({
      next: (response) => {
        this.presentToast('Simulazione avviata con successo', 'success');
        console.log('Risposta dal backend:', response);
      },
      error: (error) => {
        const errorMessage = error.error.detail || error.error.error || error.message;
        this.presentToast(`Errore durante la simulazione: ${errorMessage}`, 'danger');
      }
    });
  }

  goToBlobSimulation() {
    this.router.navigate(['/blob-simulation']);
  }
}
