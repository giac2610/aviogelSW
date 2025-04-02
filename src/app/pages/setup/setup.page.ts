import { Component, OnInit } from '@angular/core';
import { SetupAPIService, Settings } from 'src/app/services/setup-api.service';

@Component({
  selector: 'app-setup',
  templateUrl: './setup.page.html',
  styleUrls: ['./setup.page.scss'],
  standalone: false,
})
export class SetupPage implements OnInit {
  selectedMotor: string = 'motor1';
  settings: Settings[] = [];

  constructor(private configService: SetupAPIService ) { }

  ngOnInit() {
    this.loadConfig();
    
  }

  loadConfig() {
    this.configService.getSettings().subscribe((data: Settings[]) => {
      this.settings = data;
      this.test()
    });
  }

  test(){
    console.log(this.settings)
  }
}
