import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class LedService {
  private baseUrl = 'http://localhost:8000/led/control-leds/';
  private ESPurl = 'http://192.168.159.130:80/';

  constructor(private http: HttpClient) { }

  startWaveEffect() {
    const response = new Observable(observer => {
      this.http.get(this.ESPurl + 'rainbow').subscribe({
        next: (res) => {
          observer.next(res);
          observer.complete();
        },
        error: (err) => observer.error(err)
      });
    });
    console.log('Starting wave effect, response:', response);
    return response
    // console.log('Starting wave effect, response:', response);
    
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
