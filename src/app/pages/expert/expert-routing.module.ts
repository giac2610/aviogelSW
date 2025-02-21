import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';

import { ExpertPage } from './expert.page';

const routes: Routes = [
  {
    path: '',
    component: ExpertPage
  }
];

@NgModule({
  imports: [RouterModule.forChild(routes)],
  exports: [RouterModule],
})
export class ExpertPageRoutingModule {}
