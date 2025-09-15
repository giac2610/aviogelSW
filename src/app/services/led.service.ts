import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
@Injectable({
  providedIn: 'root'
})
export class LedService {
  private baseUrl = 'http://localhost:8000/led/control-leds/';
  // private ESPurl = 'http://192.168.159.130:80/';

  constructor(private http: HttpClient) { }

  startWaveEffect() {
    return this.http.post(this.baseUrl, { effect: 'wave' });
      // return this.http.get(this.ESPurl + 'rainbow');
  }
  startGreenLoading() {
    return this.http.post(this.baseUrl, { effect: 'green_loading'})
    // return this.http.get(this.ESPurl + 'greenloading');
  }

  startYellowBlink() {
    return this.http.post(this.baseUrl, { effect: 'yellow_blink'})
    // return this.http.get(this.ESPurl + 'yellowblink');
  }

  startRedStatic() {
    return this.http.post(this.baseUrl, { effect: 'red_static'})
    // return this.http.get(this.ESPurl + 'redstatic');
  }

  stopEffect() {
    return this.http.post(this.baseUrl, { effect: 'stop'})
    // return this.http.get(this.ESPurl + 'off');
  }
}
