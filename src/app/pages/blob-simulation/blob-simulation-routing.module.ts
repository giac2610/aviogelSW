import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';

import { BlobSimulationPage } from './blob-simulation.page';

const routes: Routes = [
  {
    path: '',
    component: BlobSimulationPage
  }
];

@NgModule({
  imports: [RouterModule.forChild(routes)],
  exports: [RouterModule],
})
export class BlobSimulationPageRoutingModule {}
