import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { RouteReuseStrategy } from '@angular/router';

import { IonicModule, IonicRouteStrategy } from '@ionic/angular';

import { AppComponent } from './app.component';
import { AppRoutingModule } from './app-routing.module';
import { provideHttpClient } from '@angular/common/http';
import { LongPressDirective } from './directives/long-press.directive';

@NgModule({
  declarations: [AppComponent, LongPressDirective],
  imports: [BrowserModule, IonicModule.forRoot(), AppRoutingModule],
  providers: [{ provide: RouteReuseStrategy, useClass: IonicRouteStrategy },   provideHttpClient(),],
  bootstrap: [AppComponent],
  exports: [LongPressDirective],
})
export class AppModule {}

