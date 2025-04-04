import { MotorsControlService } from './../../services/motors-control.service';
import { Component, OnInit } from '@angular/core';
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

  constructor(private configService: SetupAPIService, private toastController: ToastController, private motorsService: MotorsControlService) { }

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

  goToPosition(motor: "syringe" | "extruder" | "conveyor", distance: number) {
    const maxTravel = this.settings.motors[motor].maxTravel;

    if (this.positions[motor] + distance <= maxTravel) {
      this.positions[motor] += distance; 
      console.log(`Nuova posizione di ${motor}:`, this.positions[motor]);
      const body = {
        "motor": motor,
        "distance": distance
      }
      this.motorsService.moveMotor(body).subscribe({})
    } else {
      console.warn(`${motor} ha raggiunto il limite massimo.`);
      this.presentPositionToast( `posizione non ammessa per: ${motor} motor`)
    }
  }

  goHome(motor: "syringe" | "extruder" | "conveyor"){
    this.positions[motor] = 0;

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
