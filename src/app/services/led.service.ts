import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
@Injectable({
  providedIn: 'root'
})
export class LedService {
  private baseUrl = 'http://localhost:8000/led/control-leds/';

  constructor(private http: HttpClient) { }

  startWaveEffect() {
    return this.http.post(this.baseUrl, { effect: 'wave' });
  }
  startGreenLoading() {
    return this.http.post(this.baseUrl, { effect: 'green_loading'})
  }

  startYellowBlink() {
    return this.http.post(this.baseUrl, { effect: 'yellow_blink'})
  }

  startRedStatic() {
    return this.http.post(this.baseUrl, { effect: 'red_static'})
  }

  stopEffect() {
    return this.http.post(this.baseUrl, { effect: 'stop'})
  }
}
