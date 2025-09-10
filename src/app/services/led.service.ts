import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class LedService {
  private baseUrl = 'http://localhost:8000/led/control-leds/';
  private ESPurl = 'http://192.168.159.130:80/';

  constructor(private http: HttpClient) { }

  startWaveEffect() {
    this.http.get(this.ESPurl + 'rainbow').subscribe(response => {
      console.log('Wave effect started:', response);
      return response
    }
    , error => {
      console.error('Error starting wave effect:', error);
      return error
    });
    //
    }

  startGreenLoading() {
    return this.http.get(this.ESPurl + 'greenloading');
  }

  startYellowBlink() {
    return this.http.get(this.ESPurl + 'yellowblink');
  }

  startRedStatic() {
    return this.http.get(this.ESPurl + 'redstatic');
  }

  stopEffect() {
    return this.http.get(this.ESPurl + 'off');
  }
}
