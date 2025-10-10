import { MotorsControlService } from './../../services/motors-control.service';
import { Component, OnInit, OnDestroy, AfterViewInit, ViewChild } from '@angular/core';
import { Router } from '@angular/router';
import { IonInput, ToastController } from '@ionic/angular';
import { SetupAPIService, Settings } from 'src/app/services/setup-api.service';
import { debounceTime, Subject, Subscription } from 'rxjs';
import { LedService } from 'src/app/services/led.service';
import KioskBoard from 'kioskboard';
@Component({
    selector: 'app-setup',
    templateUrl: './setup.page.html',
    styleUrls: ['./setup.page.scss'],
    standalone: false,
})

export class SetupPage implements OnInit, OnDestroy, AfterViewInit {
    selectedMotor: string = 'none';
    selectedCameraSettings: string = 'none';
    settings!: Settings;
    testMode: boolean = false;
    isLoading = true; 
    globalGranularity: number = 1; 
    selectedPreset: string = '';
    isThreshold: boolean = false; 
    thresholdStreamUrl: string = this.configService.getThresholdStreamUrl();
    normalStreamUrl: string = 'http://localhost:8000/camera/stream/';
    selectedStream: string = 'normal';
    currentStreamUrl: string = this.normalStreamUrl; 
    

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

    @ViewChild('keyboardInput') keyboardInput!: IonInput;
    @ViewChild('numericInput') numericInput!: IonInput;

    constructor(private configService: SetupAPIService, private toastController: ToastController, private motorsService: MotorsControlService, private router: Router, private ledService: LedService) { }

    ngOnInit() {
        this.isLoading = true;
        this.loadConfig();

        // Ascolta i cambiamenti nei settaggi della camera e invia al backend
        this.cameraSettingsSubject.pipe(debounceTime(300)).subscribe((cameraSettings) => {
            this.configService.updateCameraSettings(cameraSettings).subscribe({
                next: (response) => { 
                    console.log('Impostazioni aggiornate in live:', response);
                    this.presentToast('Impostazioni camera aggiornate in live', 'success');
                },
                error: (error) => {
                    console.error('Errore durante l\'aggiornamento delle impostazioni in live:', error);
                    this.presentToast('Errore durante l\'aggiornamento delle impostazioni in live', 'danger');
                }
            });
        });

        // homing extrduer e syringe all'avvio della pagina
        this.motorsService.goHome({ motor: 'extruder' }).subscribe({
            next: (response) => {
                console.log('Risposta dal backend:', response);
                this.positions.extruder = 0; // Reset della posizione dell'extruder
            },
            error: (error) => {
                const errorMessage = error.error.detail || error.error.error || error.message;
                this.presentToast(`Errore durante il ritorno a casa dell'extruder: ${errorMessage}`, 'danger');
                console.error('Errore durante il ritorno a casa dell\'extruder:', error);
            }
        });
        this.motorsService.goHome({ motor: 'syringe' }).subscribe({
            next: (response) => {
                console.log('Risposta dal backend:', response);
                this.positions.syringe = 0; // Reset della posizione della siringa
            },
            error: (error) => {
                const errorMessage = error.error.detail || error.error.error || error.message;
                this.presentToast(`Errore durante il ritorno a casa della siringa: ${errorMessage}`, 'danger');
                console.error('Errore durante il ritorno a casa della siringa:', error);
            }
        });
    }

    ngOnDestroy() {
    }

    async ngAfterViewInit() {
        // 3. Ottieni l'elemento <input> nativo dal componente IonInput
        // Nota che getInputElement() restituisce una Promise, quindi usiamo "await"
        const nativeKeyboardInput = await this.keyboardInput.getInputElement();
        const nativeNumericInput = await this.numericInput.getInputElement();

        // 4. Inizializza KioskBoard passando direttamente l'elemento, non un selettore
        KioskBoard.run(nativeKeyboardInput, {
            keysArrayOfObjects: [
                { "0": "q", "1": "w", "2": "e", "3": "r", "4": "t", "5": "y", "6": "u", "7": "i", "8": "o", "9": "p"},
                { "0": "a", "1": "s", "2": "d", "3": "f", "4": "g", "5": "h", "6": "j", "7": "k", "8": "l", "9": "à", "10": "ò", "11": "ù", "12": "è"},
                { "0": "z", "1": "x", "2": "c", "3": "v", "4": "b", "5": "n", "6": "m", "7": ",", "8": ".", "9": "-"}
            ]
            });
        
        KioskBoard.run(nativeNumericInput, {
        // Opzioni per la tastiera numerica
        keysArrayOfObjects: [
            { "0": "1", "1": "2", "2": "3" },
            { "0": "4", "1": "5", "2": "6" },
            { "0": "7", "1": "8", "2": "9" },
            { "0": ",", "1": "0", "2": "{backspace}" }
        ]
        });
    }
    loadConfig() {
        this.configService.getSettings().subscribe((data: Settings) => {
            this.settings = data;
            // Carica x e y della camera se presenti
            if (this.settings.camera && typeof this.settings.camera.origin_x === 'number' && typeof this.settings.camera.origin_y === 'number') {
                this.cameraOrigin.x = this.settings.camera.origin_x;
                this.cameraOrigin.y = this.settings.camera.origin_y;
            }
            if (this.settings.camera.picamera_config && this.settings.camera.picamera_config.main && this.settings.camera.picamera_config.main.size) {
                this.selectedPreset = `${this.settings.camera.picamera_config.main.size[0]}x${this.settings.camera.picamera_config.main.size[1]}@${this.settings.camera.picamera_config.controls.FrameRate}`;
                this.isLoading = false;
            }

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

    closeCamera(){
        console.log('Chiusura della camera');
        this.selectedCameraSettings = 'none';
        this.currentStreamUrl = '';
        this.configService.deInitializeCamera().subscribe({
            next: (response) => {
                console.log('Camera de-inizializzata:', response);
                this.presentToast('Camera de-inizializzata con successo', 'success');
            },
            error: (error) => {
                console.error('Errore durante la de-inizializzazione della camera:', error);
                this.presentToast('Errore durante la de-inizializzazione della camera', 'danger');
            }
        });
    }

    onPresetChange() {
        const [resolution, fps] = this.selectedPreset.split('@');
        const [width, height] = resolution.split('x').map(Number);

        this.settings.camera.picamera_config.main.size[0] = width;
        this.settings.camera.picamera_config.main.size[1] = height;
        this.settings.camera.picamera_config.controls.FrameRate = Number(fps);

        this.onCameraSettingChange();
    }

    goToPosition(motor: "syringe" | "extruder" | "conveyor", distance: number) {
        const maxTravel = this.settings.motors[motor].maxTravel;

        if (maxTravel < 0 || this.positions[motor] + distance <= maxTravel) {
            this.positions[motor] += distance; 
            const targets = { [motor]: distance }; // Correctly structure the targets object
            console.log("request: ", targets);
            this.configService.moveMotor(targets).subscribe({
                next: (response) => {
                    
                    this.presentToast(`Successo: ${response.status} - Target: ${JSON.stringify(response.targets)}`, 'success');
                    console.log('Risposta dal backend:', response);
                },
                error: (error) => {
                    
                    const errorMessage = error.error.detail || error.error.error || error.message;
                    this.presentToast(`Errore: ${errorMessage}`, 'danger');
                }
            });
        } else {
            this.presentToast(`Posizione non ammessa per il motore ${motor}`, 'danger');
        }
    }
    
    goHome(motor: "syringe" | "extruder" | "conveyor"){
        const targets = { "motor": motor };
        this.motorsService.goHome(targets).subscribe({
            next: (response) => {
                this.presentToast(`Il motore ${motor} è tornato a casa con successo`, 'success');
                console.log('Risposta dal backend:', response);
                this.positions[motor] = 0;
            },
            error: (error) => {
                const errorMessage = error.error.detail || error.error.error || error.message;
                this.presentToast(`Errore durante il ritorno a casa del motore ${motor}: ${errorMessage}`, 'danger');
                console.error('Errore durante il ritorno a casa del motore:', error);
            }
        });


    }

    stopMotors() {
        this.configService.stopMotors().subscribe({
            next: (response) => {
                this.presentToast(`Successo: ${response.status}`, 'success');
            },
            error: (error) => {
                this.presentToast(`Errore: ${error.error.detail || error.message}`, 'danger');
            }
        });
    }

    updateMotorHertz(motor: "syringe" | "extruder" | "conveyor") {
        const motorSettings = this.settings.motors[motor];
        motorSettings.hertz = (motorSettings.maxSpeed * motorSettings.stepOneRev * motorSettings.microstep) / motorSettings.pitch;
    }

    saveSettings() {
        ["syringe", "extruder", "conveyor"].forEach((motor) => {
            const motorSettings = this.settings.motors[motor as "syringe" | "extruder" | "conveyor"];
            motorSettings.hertz = (motorSettings.maxSpeed * motorSettings.stepOneRev * motorSettings.microstep) / motorSettings.pitch;
            motorSettings.stepsPerMm = (motorSettings.stepOneRev * motorSettings.microstep) / motorSettings.pitch;
        });

        console.log("Dati inviati al backend:", this.settings);
        this.configService.updateSettings(this.settings).subscribe({
            next: (response) => {
                console.log('Impostazioni salvate:', response);
                this.presentToast('Impostazioni salvate con successo', 'success');
            },
            error: (error) => {
                this.presentToast('Errore durante il salvataggio delle impostazioni', 'danger');
            }
        });
        this.motorsService.updateSettings({}).subscribe({
            next: (response) => {
                console.log('Impostazioni motori aggiornate:', response);
                this.presentToast('Impostazioni motori aggiornate con successo', 'success');
            },
            error: (error) => {
                console.error('Errore durante l\'aggiornamento delle impostazioni motori:', error);
                this.presentToast('Errore durante l\'aggiornamento delle impostazioni motori', 'danger');
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

    async presentToast(message: string, color: string = 'success', duration=1400) {
        const toast = await this.toastController.create({
            message: message,
            duration: duration,
            icon: color === 'success' ? 'checkmark-circle' : 'alert-circle',
            color: color
        });
        await toast.present();
    }

    updateStreamUrl() {
        switch (this.selectedStream) {
            case 'fixed':
                this.currentStreamUrl = this.configService.getFixedPerspectiveStreamUrl(); // Modalità fissa con keyframe
                break
            case 'threshold':
                this.currentStreamUrl = this.configService.getThresholdStreamUrl(); // Modalità threshold con keyframe
                break;
            case 'normal':
            default:
                this.currentStreamUrl = this.configService.getNormalStreamUrl(); // Modalità normale con keyframe
                break;
        }
        console.log(`Stream URL aggiornato a: ${this.currentStreamUrl}`);
    }

    onImageError() {
        console.error("Errore durante il caricamento dello streaming.");
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

    goToImageCalGen(){
        this.configService.saveFrameCalibration().subscribe({
            next: (response) => {
                this.presentToast('Frame calibration saved successfully', 'success');
                console.log('Response from backend:', response);
                
            },
            error: (error) => {
                const errorMessage = error.error.detail || error.error.error || error.message;
                this.presentToast(`Error saving frame calibration: ${errorMessage}`, 'danger');
                console.log('Response from backend:', errorMessage);
            }
        });
    }

    calibrateCamera() {
        this.configService.calibrateCamera().subscribe({
            next: (response) => {
                this.presentToast('Calibrazione della camera avviata con successo', 'success');
                console.log('Risposta dal backend:', response);
            },
            error: (error) => {
                const errorMessage = error.error.detail || error.error.error || error.message;
                this.presentToast(`Errore durante la calibrazione della camera: ${errorMessage}`, 'danger');
                console.log('Risposta dal backend:', errorMessage);
            }
        });
    }

    resetCameraCalibration() {
        this.configService.resetCameraCalibration().subscribe({
            next: (response) => {

                this.presentToast('Calibrazione della camera resettata con successo', 'success');
                console.log('Risposta dal backend:', response);
            },
            error: (error) => {
                const errorMessage = error.error.detail || error.error.error || error.message;
                this.presentToast(`Errore durante il reset della calibrazione della camera: ${errorMessage}`, 'danger');
                console.log('Risposta dal backend:', errorMessage);
            }
        });
    }

    setFixedPerspective(){
        this.configService.setFixedPerspectiveView().subscribe({
            next: (response) => {
                this.presentToast('Vista fissa impostata con successo', 'success');
                console.log('Risposta dal backend:', response);
            }
            , error: (error) => {
                const errorMessage = error.error.detail || error.error.error || error.message;
                this.presentToast(`Errore durante l'impostazione della vista fissa: ${errorMessage}`, 'danger');
            }
        });
    }

    goToBlobSimulation() {
        this.router.navigate(['/blob-simulation']);
    }

    getIpAddress() {
        this.configService.getIpAddress().subscribe({
            next: (data) => {
                console.log('Dati ricevuti dal backend:', data);
                const ip = data.ip_address;
                this.presentToast(`Indirizzo IP del server: ${ip}`, 'success', 5000);
                console.log('Indirizzo IP del server:', ip);
            },
            error: (error) => {
                this.presentToast('Errore nel recuperare l\'indirizzo IP del server', 'danger');
                console.error('Errore nel recuperare l\'indirizzo IP del server:', error);
            }
        });
    }

}
