import { ComponentFixture, TestBed } from '@angular/core/testing';
import { BlobSimulationPage } from './blob-simulation.page';

describe('BlobSimulationPage', () => {
  let component: BlobSimulationPage;
  let fixture: ComponentFixture<BlobSimulationPage>;

  beforeEach(() => {
    fixture = TestBed.createComponent(BlobSimulationPage);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
